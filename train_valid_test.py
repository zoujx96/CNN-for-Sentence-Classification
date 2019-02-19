from collections import defaultdict
import argparse
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
import numpy as np
import math

# Model architecture
class CNN_Model(torch.nn.Module):
    def __init__(self, emb_size, num_filters, window_size, ntags):
        super(CNN_Model, self).__init__()
        # Define model parameters
        self.num_filters = num_filters
        # Define layers
        self.conv_1d = torch.nn.Conv1d(in_channels=emb_size, out_channels= \
            self.num_filters, kernel_size=window_size, stride=1, padding=0, \
            dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        self.pool_1d = torch.nn.AdaptiveMaxPool1d(1, return_indices=False)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.projection_layer = torch.nn.Linear(in_features=self.num_filters, \
            out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words):
        h = self.conv_1d(words)
        h = self.relu(h)
        h = self.pool_1d(h)
        h = h.view(-1, self.num_filters)
        h = self.dropout(h)
        out = self.projection_layer(h)
        return out

# Divide sentences with the same length into clusters
def data_cluster(dataset):
    sentences = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]
    sentence_lengths = np.array([len(x) for x in sentences])
    sorted_index = np.argsort(sentence_lengths)
    sentence_set = []
    label_set = []
    sentence_cluster = []
    label_cluster = []
    prev_length = len(sentences[sorted_index[0]])
    for index in sorted_index:
        sentence = sentences[index]
        label = labels[index]
        if len(sentence) != prev_length:
            sentence_set.append(np.array(sentence_cluster))
            label_set.append(np.array(label_cluster))
            sentence_cluster = []
            label_cluster = []
            sentence_cluster.append(sentence)
            label_cluster.append(label)
            prev_length = len(sentence)
        else:
            sentence_cluster.append(sentence)
            label_cluster.append(label)
    return sentence_set, label_set

# Load data and convert the words into their embeddings
def read_dataset(filename, w_v_model, t2i):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            words = words.split(" ")
            yield (np.array([w_v_model[x] if x in w_v_model.keys() else \
                w_v_model["UNK"] for x in words]), t2i[tag])

# Load pre-trained word embeddings
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def train(model, device, train_sentence_set, train_label_set, \
    dev_sentence_set, dev_label_set, num_epoch, batch_size):
    # Set up the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
        mode='min',factor=0.5, patience=2, verbose=True)

    # Define saver
    best_dev_acc = 0.0
    model_saver = 'model/cnn_model.pt'

    # Plot variables
    epoch_plot = []
    train_loss_plot = []
    dev_loss_plot = []

    # Start training
    for epoch in range(num_epoch):
        train_loss = train_one_epoch(model, device, optimizer, \
            train_sentence_set, train_label_set, batch_size)
        dev_loss, dev_acc = validate(model, device, dev_sentence_set, \
            dev_label_set, epoch, batch_size)

        print('\rTrain Epoch {:>2}: training loss is {:.4f}, validation loss is {:.4f} and validation accuracy is {:.4f}'. \
            format(epoch, train_loss, dev_loss, dev_acc))

        scheduler.step(dev_loss)

        if dev_acc > best_dev_acc:
            torch.save(model.state_dict(), model_saver)
            print('Train Epoch{:>3}: validation accuracy improve from {:.4f} to {:.4f}\n'. \
                format(epoch, best_dev_acc, dev_acc))
            best_dev_acc = dev_acc
        else:
            print('Train Epoch{:>3}: validation accuracy did not improve from {:.4f}\n'. \
                format(epoch, best_dev_acc))

        epoch_plot.append(epoch)
        train_loss_plot.append(train_loss)
        dev_loss_plot.append(dev_loss)

    # Plot learning curve
    plot_learning_curve(epoch_plot, train_loss_plot, dev_loss_plot)

    # Load the model with the highest validation accuracy
    model.load_state_dict(torch.load(model_saver))

    return model

def train_one_epoch(model, device, optimizer, train_sentence_set, \
    train_label_set, batch_size):
    # Prepare for training
    model.train()
    train_loss = 0
    total_num = 0

    # Shuffle clusters
    num_cluster = len(train_label_set)
    cluster_index = np.arange(num_cluster)
    np.random.shuffle(cluster_index)
    iterate_how_many_cluster_until_now = 1
    
    for index in cluster_index:
        current_sentence_cluster = train_sentence_set[index]
        current_label_cluster = train_label_set[index]

        num_sample_in_cluster = len(current_label_cluster)
        num_batch = math.ceil(num_sample_in_cluster / batch_size)
        
        # Shuffle sentences within one cluster
        sample_index = np.arange(num_sample_in_cluster)
        np.random.shuffle(sample_index)
        shuffled_sentences = current_sentence_cluster[sample_index]
        shuffled_labels = current_label_cluster[sample_index]

        # Iterate batches in one cluster
        for i in range(num_batch):
            if i == num_batch - 1:
                sentences = shuffled_sentences[i * batch_size: \
                    num_sample_in_cluster]
                labels = shuffled_labels[i * batch_size:num_sample_in_cluster]
            else:
                sentences = shuffled_sentences[i * batch_size:(i + 1) * \
                    batch_size]
                labels = shuffled_labels[i * batch_size:(i + 1) * batch_size]
            
            sentence_tensor = torch.from_numpy(sentences).float(). \
                permute(0, 2, 1).to(device)
            label_tensor = torch.from_numpy(labels).to(device)
            scores = model(sentence_tensor)
            loss = F.cross_entropy(scores, label_tensor)
            train_loss += loss.item() * len(label_tensor)
            total_num += len(label_tensor)

            # Do back-prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        iterate_how_many_cluster_until_now += 1

    train_loss = train_loss / total_num

    return train_loss

def validate(model, device, dev_sentence_set, dev_label_set, epoch, batch_size):
    # Prepare for validation
    model.eval()
    dev_loss = 0
    total_num = 0
    correct_num = 0
    with torch.no_grad():
        num_cluster = len(dev_label_set)
        cluster_index = np.arange(num_cluster)

        for index in cluster_index:
            current_sentence_cluster = dev_sentence_set[index]
            current_label_cluster = dev_label_set[index]

            num_sample_in_cluster = len(current_label_cluster)
            num_batch = math.ceil(num_sample_in_cluster / batch_size)
            
            for i in range(num_batch):
                if i == num_batch - 1:
                    sentences = current_sentence_cluster[i * batch_size: \
                        num_sample_in_cluster]
                    labels = current_label_cluster[i * batch_size: \
                        num_sample_in_cluster]
                else:
                    sentences = current_sentence_cluster[i * batch_size: \
                        (i + 1) * batch_size]
                    labels = current_label_cluster[i * batch_size:(i + 1) \
                        * batch_size]

                sentence_tensor = torch.from_numpy(sentences).float(). \
                    permute(0, 2, 1).to(device)
                label_tensor = torch.from_numpy(labels).to(device)
                scores = model(sentence_tensor)
                dev_loss += F.cross_entropy(scores, label_tensor).item() * \
                    len(label_tensor)
                pred = torch.argmax(scores, dim=1)
                correct_num += (pred == label_tensor).sum().item()
                total_num += len(label_tensor)

        dev_acc = correct_num / total_num
        dev_loss = dev_loss / total_num

    return dev_loss, dev_acc

def predict_tag(model, device, file_name, data, tag_dict):
    file = open(file_name, "a")
    for item in data:
        sentence_tensor = torch.from_numpy(item[0]).permute(1, 0). \
            unsqueeze(0).float().to(device)
        scores = model(sentence_tensor)
        pred = torch.argmax(scores, dim=1).item()
        file.write(tag_dict[pred] + '\n')

    file.close()

def plot_learning_curve(x, y1, y2):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, 'g', label="training loss")
    ax1.legend(loc='upper right')
    ax1.set_ylim([0.5, 1])
    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'b', label="validation loss")
    ax2.legend(loc='upper left')
    ax2.set_ylim([0.5, 1])
    plt.xlabel(r"epoch", fontsize=16)
    plt.savefig("figure/learning_curve.png")

def main(args):
    # Set the random seed manually for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Switch backend
    plt.switch_backend('Agg')

    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained word embeddings and data
    t2i = defaultdict(lambda: len(t2i))

    w_v = load_vectors("wiki-news-300d-1M.vec")
    print("Vocabulary loaded!!")
    train_set = list(read_dataset("dataset/topicclass_train.txt", w_v, t2i))
    dev_set = list(read_dataset("dataset/topicclass_valid.txt", w_v, t2i))
    test_set = list(read_dataset("dataset/topicclass_test.txt", w_v, t2i))

    print("Dataset constructed!")
    print("Training set: %s samples" % len(train_set))
    print("Validation set: %s samples" % len(dev_set))
    print("Test set: %s samples" % len(test_set))

    # Cluster sentences
    train_sentence_set, train_label_set = data_cluster(train_set)
    dev_sentence_set, dev_label_set = data_cluster(dev_set)

    print("Cluster_constructed!")
    print("Training set: %s clusters" % len(train_sentence_set))
    print("Validation set: %s clusters" % len(dev_sentence_set))

    # Define the model parameters
    EMB_SIZE = 300
    WIN_SIZE = 3
    BATCH_SIZE = 64
    NUM_FILTER = 64
    NUM_TAG = 16
    NUM_EPOCH = 20

    # Initialize the model
    model = CNN_Model(EMB_SIZE, NUM_FILTER, WIN_SIZE, NUM_TAG).to(device)

    # Train and validate the model
    best_model = train(model, device, train_sentence_set, train_label_set, \
        dev_sentence_set, dev_label_set, NUM_EPOCH, BATCH_SIZE)

    # Reverse the dictionary
    i2t = dict(zip(t2i.values(), t2i.keys()))

    # Prepare for prediction
    best_model.eval()

    # Predict the topics for the validation set and test set
    predict_tag(best_model, device, "predict_valid.txt", dev_set, i2t)
    predict_tag(best_model, device, "predict_test.txt", test_set, i2t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Validating, and Testing the model for MNIST')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    arguments = parser.parse_args()
    main(arguments)




