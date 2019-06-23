import ChI
import numpy as np
import csv
from os import listdir
from os.path import isfile, join


def soft_max(samples):
    # Normalizes each sample w.r.t. each sample (soft max)
    for i in range(0, samples.shape[0]):  # for each sample
        for j in range(0, samples.shape[1]):  # for each network
            samples[i,j,:] = np.exp(samples[i, j, :]) / sum(np.exp(samples[i, j, :]))

    return [samples]



# Specify Network Path
network_path = 'C:\\Users\\MINDFUL\\PycharmProjects\\fusion\\data\\'
cross_val_path = 'C:\\Users\\MINDFUL\\PycharmProjects\\fusion\\data\\cross_val\\'

# set up variables
image_names = []

# This is how we'll get all of the different inputs. Each csv file needs to follow the correct format
csv_files = [f for f in listdir(network_path) if isfile(join(network_path, f))]
cross_val = [f for f in listdir(cross_val_path) if isfile(join(cross_val_path, f))]
num_nets = csv_files.__len__()

# Create dictionary to store samples
data = dict.fromkeys(csv_files)

# Read in all of the csv data into a dictionary
densities = []
for file in cross_val:
    csv_data = []
    data_info = np.genfromtxt((cross_val_path + '\\' + file), usecols=(1), skip_header=True,dtype="f", delimiter=',')
    densities.append(np.mean(data_info))

densities = np.asarray(densities)
first_net = 1 # this is a flag.
for file in csv_files:
    csv_data = []
    data_info = np.genfromtxt((network_path + '\\' + file), usecols=(1, 2, 3, 4), dtype="|U", delimiter=',')
    confidence_vectors = np.genfromtxt((network_path + '\\' + file), delimiter=';')

    for line in range(0, data_info.__len__()):
        if first_net:
            image_names.append(data_info[line,0])
        csv_data.append(np.hstack((data_info[line,:-1], data_info[line,3].partition(';')[0], confidence_vectors[line, 1:])))

    first_net = 0
    data[file] = csv_data


# How many classes are there?
num_classes = confidence_vectors.shape[1]# Assuming the first 4 columns are 'image	y_true	confidence	y_pred'

    # Now I need to build the samples and their corresponding labels
    # There will be the same number of ChI's as there are classes(L0
    # One ChI per class, so each one sample will turn into L samples
num_samples = data_info.__len__()
samples = np.zeros([num_samples, csv_files.__len__(), num_classes])
label = np.zeros([num_samples, num_classes])

print('--Formatting Samples--')

############################################################################################
# Get samples for each ChI - samples is a [num_classes x num_inputs x num_samples]
############################################################################################
for i, img in enumerate(image_names):        # For each image
    for n, net in enumerate(csv_files):      # For each net
        temp = data[net]
        for l in temp: #
            if img == l[0]:
                for c in range(0, num_classes):
                    samples[i, n, c] = l[c+3]
                    if str(c) == l[1]:
                        label[i, c] = 1
                    else:
                        label[i, c] = 0


##############################################
# Start Training and Testing
##############################################
print('--Starting Training and Testing')


train_samples = samples.copy()
train_labels  = label

test_samples = samples.copy()
test_labels = label
##############################################
# Normalize Training Data & Testing Data
##############################################
[train_samples] = soft_max(train_samples)
[test_samples] = soft_max(test_samples)

print('--Training--')
##############################################
# Train the ChI(s)
##############################################

CHIs = []
for j in range(0, num_classes):
    CHIs.append(ChI.ChoquetIntegral())

for j, chi in enumerate(CHIs):
    print('Class ChI {}'.format(j))
    tr = chi.train_chi_sugeno(densities)
    print(chi.fm)


print('--Testing--')
##############################################
# Test the ChI(s)
##############################################
exper_out, known_out = [], []
dec = []
for j in range(0, test_samples.shape[0]): # for each data point
    out = []
    for k, chi in enumerate(CHIs): # for each ChI
        test_sample = np.transpose(test_samples[j, :, k])
        test_label = np.argmax(test_labels[j, :])
        out.append(chi.chi_sugeno(test_sample))
    out = np.asarray(out)
    exper_out.append(np.argmax(out))
    known_out.append(test_label)
for i in range(0, exper_out.__len__()):
    if exper_out[i] == known_out[i]:
        dec.append(1)
    else:
        dec.append(0)
with open('Results{}.csv'.format(i), 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for k in range(0, exper_out.__len__()):
        spamwriter.writerow([exper_out[k], known_out[k]])

acc = np.sum(dec) / dec.__len__()
print(acc)

