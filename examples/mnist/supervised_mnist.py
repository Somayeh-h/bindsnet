import os
import sys 
import torch
import numpy as np
import argparse
from pathlib import Path 
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm
from data import *


from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_assignments,
    plot_performance,
    plot_weights,
    plot_spikes,
    plot_voltages,
)


def save_connections(connections, interval=''):
    '''
    Saves a zip of presynaptic and postsynaptic neuron connection indices and their corresponding weight value 
    '''

    print('save connections')

    for connection in connections: 

        connection_name = connection[0] + connection[1]

        connection_weights = network.connections[(connection[0], connection[1])].w.cpu().detach().numpy() 

        print(weights_path + connection_name + interval) 

        np.save( weights_path+connection_name+interval, connection_weights) 



parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=400)
parser.add_argument("--n_train", type=int, default=25)
parser.add_argument("--n_test", type=int, default=25)
parser.add_argument("--n_clamp", type=int, default=1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=350)
parser.add_argument("--dt", type=int, default=1)
parser.add_argument("--intensity", type=float, default=4) # 32
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--epochs", type=int, default=25)
parser.set_defaults(plot=True, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_train = args.n_train

n_test = args.n_test
n_clamp = args.n_clamp
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
device_id = args.device_id
epochs = args.epochs 


# Sets up GPU use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False


data_path = './outputs/outputs_ne{}_L{}/'.format(n_neurons, n_test)
weights_path = './weights/weights_ne{}_L{}/'.format(n_neurons, n_test)

Path(data_path).mkdir(parents=True, exist_ok=True)
Path(weights_path).mkdir(parents=True, exist_ok=True)

f = open(data_path + "log.out", 'w')
sys.stdout = f 


torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

print('A', sys.version)
print('B', torch.__version__)
print('C', torch.cuda.is_available())
print('D', torch.backends.cudnn.enabled)
device = torch.device('cuda')
print('E', torch.cuda.get_device_properties(device))
print('F', torch.tensor([1.0, 2.0]).cuda())


org_data_path = ['./../data/nordland/'] 
train_data_path = [org_data_path[0] + 'spring/', org_data_path[0] + 'fall/'] 
test_data_path =  [org_data_path[0] + 'summer/']

imWidth = 28
imHeight = 28 

if not train:
    update_interval = n_test

n_classes = n_test
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
per_class = int(n_neurons / n_classes)

num_examples = len(train_data_path) * n_train * epochs

# Build Diehl & Cook 2015 network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    nu=[1e-10, 1e-3],  # 0.711
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time, device=device)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time, device=device)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]),
)

frames, encoded_frames = processImageDataset(train_data_path, "train", imWidth, imHeight, time=time, dt=dt, desired_num_labels=n_train, intensity=intensity)
frames = np.array(frames)

y = list(range( int(frames.shape[0] / (len(train_data_path)/len(org_data_path)) ) ))
y = y * ( int((len(train_data_path)) / len(org_data_path)) )
y = [ [y[i]] for i in range(len(y))]

training = {'x': encoded_frames, 'y': y}


frames, encoded_frames_t = processImageDataset(test_data_path, "test", imWidth, imHeight, time=time, dt=dt, desired_num_labels=n_test, intensity=intensity)
frames = np.array(frames)

y = list(range( frames.shape[0] ))
y_t = [ [y[i]] for i in range(len(y))]

testing = {'x': encoded_frames_t, 'y': y_t}


# # http://man.hubwiz.com/docset/torchvision.docset/Contents/Resources/Documents/datasets.html
# imagenet_data = torchvision.datasets.ImageFolder(root='./../data/nordland/spring/')

# dataloader_vpr = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons, device=device)

# Neuron assignments and spike proportions.
assignments = -torch.ones_like(torch.Tensor(n_neurons), device=device)
proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Labels to determine neuron assignments and spike proportions and estimate accuracy
labels = torch.empty(update_interval, device=device)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

# Train the network.
print("Begin training.\n")

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes = None
voltage_ims = None

pbar = tqdm(total=num_examples) 
# for (i, datum) in enumerate(dataloader): n_train
for i in range(num_examples):
    if i > num_examples:
        break

    image = training['x'][i%n_train]                            # datum["encoded_image"]
    label = torch.Tensor(training['y'][i%n_train])              # datum["label"]

    if i % update_interval == 0 and i > 0:
        # Get network predictions.
        all_activity_pred = all_activity(spike_record, assignments, n_classes)
        proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_classes)

        # Compute network accuracy according to available classification strategies.
        accuracy["all"].append(100 * torch.sum(labels.long() == all_activity_pred).item() / update_interval)
        accuracy["proportion"].append(100 * torch.sum(labels.long() == proportion_pred).item() / update_interval)

        print("\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)" % (accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"])))

        print("Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n" % (accuracy["proportion"][-1], np.mean(accuracy["proportion"]), np.max(accuracy["proportion"])))

        # Assign labels to excitatory layer neurons.
        assignments, proportions, rates = assign_labels(spike_record, labels, n_classes, rates)

        spike_record_np = spike_record.cpu().detach().numpy()
        np.save(data_path + "spike_record{}".format(i), spike_record_np)


    # Add the current label to the list of labels for this update_interval
    labels[i % update_interval] = label[0]

    # Run the network on the input.
    choice = np.random.choice(int(n_neurons / n_classes), size=n_clamp, replace=False)
    clamp = {"Ae": per_class * label.long() + torch.Tensor(choice).long()}
    if gpu:
        inputs = {"X": image.cuda().view(int(time/dt), 1, 1, 28, 28)}
    else:
        inputs = {"X": image.view(int(time/dt), 1, 1, 28, 28)}
    network.run(inputs=inputs, time=time, clamp=clamp)

    # Get voltage recording.
    exc_voltages = exc_voltage_monitor.get("v")
    inh_voltages = inh_voltage_monitor.get("v")

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)

    if i % update_interval == 0 and i > 0:
        save_connections(network.connections, str(i))


    # Optionally plot various simulation information.
    if plot:
        inpt = inputs["X"].view(int(time/dt), 784).sum(0).view(28, 28)
        input_exc_weights = network.connections[("X", "Ae")].w
        square_weights = get_square_weights(input_exc_weights.view(784, n_neurons), n_sqrt, 28)
        square_assignments = get_square_assignments(assignments, n_sqrt)
        voltages = {"Ae": exc_voltages, "Ai": inh_voltages}

        inpt_axes, inpt_ims = plot_input(image.sum(1).view(28, 28), inpt, label=label, axes=inpt_axes, ims=inpt_ims)
        
        spike_ims, spike_axes = plot_spikes({layer: spikes[layer].get("s").view(time, 1, -1) for layer in spikes}, ims=spike_ims, axes=spike_axes)


        weights_im = plot_weights(square_weights, im=weights_im, wmax=1, save="weights.png")
        assigns_im = plot_assignments(square_assignments, im=assigns_im, save="assignments.png")
        perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax, save="performance.png")
        voltage_ims, voltage_axes = plot_voltages(voltages, ims=voltage_ims, axes=voltage_axes)

        plt.pause(1e-8)

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Train progress: ")
    pbar.update()


spike_record_np = spike_record.cpu().detach().numpy()
np.save(data_path + "resultPopVecs{}".format(num_examples), spike_record_np)

spike_record_last = spike_record_np[:, -1, :]
np.save(data_path + "resultPopVecs{}_last".format(num_examples), spike_record_np)

spike_record_mean = spike_record_np.sum(axis=1) / spike_record_np.shape[1]
np.save(data_path + "resultPopVecs{}_mean".format(num_examples), spike_record_np)

spike_record_sum = spike_record_np.sum(axis=1) 
np.save(data_path + "resultPopVecs{}_sum".format(num_examples), spike_record_np)

labels_np = labels.cpu().detach().numpy()
np.save(data_path + "inputNumbers{}".format(num_examples), labels_np)

print("Progress: %d / %d \n" % (num_examples, num_examples))
print("Training complete.\n")




print("Testing....\n")

# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)])
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros(1, time, n_neurons, device=device) # int(time / dt)
spike_records = torch.zeros(n_test, time, n_neurons, device=device) 

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)

pbar = tqdm(total=n_test)
# for step, batch in enumerate(test_dataset):
for step in range(n_test):
    if step > n_test:
        break
    # Get next input sample.

    image = testing['x'][step].view([350, 1, 28, 28])                      #  batch["encoded_image"]       
    label = testing['y'][step][0]       # batch["label"]   

    inputs = {"X": image.view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()
    spike_records[step] = spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(label, device=device)

    # Get network predictions.
    all_activity_pred = all_activity(spikes=spike_record, assignments=assignments, n_labels=n_classes)

    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())

    accuracy["proportion"] += float(torch.sum(label_tensor.long() == proportion_pred).item())

    network.reset_state_variables()  # Reset state variables.

    pbar.set_description_str( f"Accuracy: {(max(accuracy['all'] ,accuracy['proportion'] ) / (step+1)):.3}" )
    pbar.update()


spike_record_np = spike_records.cpu().detach().numpy()
np.save(data_path + "resultPopVecs{}".format(n_test), spike_record_np)

labels_np = [x[0] for x in testing['y']]
np.save(data_path + "inputNumbers{}".format(n_test), labels_np)


print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))


print("Testing complete.\n")
