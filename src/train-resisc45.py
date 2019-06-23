# pylint: disable=no-member
import os, sys
from pathlib import Path
import torch
import torchvision
import torchvision.models as models
import psycopg2
from PIL import Image
from tqdm import tqdm

# Imagery location
DATASET = '/mnt/datasets/IEEE/RESISC45'

# Output space
OUTPUT = '/mnt/datasets/IEEE/RESISC45'

# Metadata location
METADATA = 'dbname=resisc45 user=postgres'

# Choose a base partition below
BASE = {0:'active', 1:'down20', 2:'down10', 3:'down1'}[0]

# Specify number of classes
NUM_CLASS = 45

# Specify number of epochs per x-fold
EPOCHS = 15

# Specify batch size for training and testing
BATCH_SIZE = 32

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, metadata, transform):
        self.dataset = Path(dataset)
        self.transform = transform
        self.metadata = metadata

    def __getitem__(self, idx):
        frame_id, image, class_label = self.metadata[idx]
        fname = str(self.dataset / image)
        return dict(frame_id=frame_id, image=self.transform(Image.open(fname).convert("RGB")), label=class_label)

    def __len__(self):
        return len(self.metadata)

transform_pipe = torchvision.transforms.Compose([
    torchvision.transforms.Resize(
        size=(299, 299)
    ),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def xval_metadata(partition, test=False):
    print('Loading (xval)', partition, '=>', ('test' if test else 'train'))
    with psycopg2.connect(METADATA) as conn:
        with conn.cursor() as cur:
            partition, fold = partition.split('::')
            if test:
                cur.execute("select id, image, class_label from frame where partitions -> %s = %s AND partitions ? %s", (partition, fold, BASE))
            else:
                cur.execute("select id, image, class_label from frame where partitions ? %s AND partitions -> %s != %s AND partitions ? %s", (partition, partition, fold, BASE))
            return cur.fetchall()

def train(model, epochs, fname_save, xval_fold=None, lr=1e-3, is_inception=False):
    train_data = ImageDataset(
        dataset=DATASET,
        metadata=xval_metadata(xval_fold or 'traintest', test=False),
        transform=transform_pipe
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_data = ImageDataset(
        dataset=DATASET,
        metadata=xval_metadata(xval_fold or 'traintest', test=True),
        transform=transform_pipe
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE
    )

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for i in range(epochs):
        model.train()
        ###################
        #     TRAINING    #
        ###################
        samples = 0
        loss_sum = 0
        true_sum = 0
        for batch in tqdm(train_loader):
            X = batch["image"].cuda()
            labels = batch["label"].cuda()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if is_inception:
                    # ref: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
                    y, aux_outputs = model(X)
                    loss1 = criterion(y, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4*loss2
                else:
                    y = model(X)
                    loss = criterion(y, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * X.shape[0]
                samples += X.shape[0]
                num_true = torch.sum(torch.argmax(y, 1) == labels)
                true_sum += num_true

        epoch_acc = float(true_sum) / float(samples)
        epoch_loss = float(loss_sum) / float(samples)
        print("epoch: {} - {} loss: {}, acc: {}".format(i + 1, "train", epoch_loss, epoch_acc))

        ###################
        #     TESTING     #
        ###################
        model.eval()
        samples = 0
        loss_sum = 0
        true_sum = 0
        for batch in test_loader:
            X = batch["image"].cuda()
            labels = batch["label"].cuda()
            with torch.set_grad_enabled(False):
                y = model(X)
                loss = criterion(y, labels)
                loss_sum += loss.item() * X.shape[0]
                samples += X.shape[0]
                num_true = torch.sum(torch.argmax(y, 1) == labels)
                true_sum += num_true

        epoch_acc = float(true_sum) / float(samples)
        epoch_loss = float(loss_sum) / float(samples)
        print("epoch: {} - {} loss: {}, acc: {}".format(i + 1, "test", epoch_loss, epoch_acc))

        ###################
        #     SAVING      #
        ###################
        torch.save(model.state_dict(), fname_save)

def get_labels():
    with psycopg2.connect(METADATA) as conn:
        with conn.cursor() as cur:
            cur.execute("select label_name, label_id from class_label")
            return dict(cur.fetchall())

def xval_per_class_metadata(partition, class_label, test=False):
    print(f'Loading (xval.{class_label})', partition, '=>', ('test' if test else 'train'))
    with psycopg2.connect(METADATA) as conn:
        with conn.cursor() as cur:
            partition, fold = partition.split('::')
            if test:
                cur.execute("select id, image, class_label from frame where class_label = %s AND partitions -> %s = %s AND partitions ? %s", (class_label, partition, fold, BASE))
            else:
                cur.execute("select id, image, class_label from frame where class_label = %s AND partitions ? %s AND partitions -> %s != %s AND partitions ? %s", (class_label, partition, partition, fold, BASE))
            return cur.fetchall()

def save_inference(cur, model_name, xval_fold, frame_ids, y_pred, y_vec):
    import psycopg2.extras
    model_id = '{}.{}'.format(model_name, xval_fold)
    sql = 'INSERT INTO inference (frame_id, model_id, prediction) values %s ON CONFLICT DO NOTHING;'
    values = []
    for frame_id, i_pred, i_vec in zip(frame_ids, y_pred, y_vec):
        values.append((frame_id, model_id, ','.join(['"y_pred"=>{}'.format(i_pred), '"y_vec"=>"{}"'.format(';'.join(map(str, i_vec)))])))
    psycopg2.extras.execute_values(cur, sql, values)

def save_metric(cur, model_name, xval_fold, class_label, acc):
    model_id = '{}.{}'.format(model_name, xval_fold)
    sql = 'INSERT INTO xval_metrics (model_id, subset_id, partition_name, fold_name, metrics) values (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING;'
    partition, fold = xval_fold.split('::')
    cur.execute(sql, (model_id, class_label, partition, fold, ('"acc"=>%f'%acc)))

def xval_evaluate(model, model_name, xval_fold):
    """ Evaluate a trained model and generate per-class metrics """
    conn = psycopg2.connect(METADATA)
    cur = conn.cursor()
    for class_label in get_labels().values():
        test_data = ImageDataset(dataset=DATASET, metadata=xval_per_class_metadata(xval_fold, class_label, test=True), transform=transform_pipe)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
        model.eval()
        samples = 0
        true_sum = 0
        for batch in test_loader:
            frame_ids = batch["frame_id"]
            X = batch["image"].cuda()
            labels = batch["label"].cuda()
            with torch.set_grad_enabled(False):
                y = model(X)
                samples += X.shape[0]
                y_pred = torch.argmax(y, 1)
                num_true = torch.sum(y_pred == labels)
                true_sum += num_true
                save_inference(cur, model_name, xval_fold, frame_ids.tolist(), y_pred.cpu().tolist(), y.cpu().tolist())
        epoch_acc = float(true_sum) / float(samples)
        print("{}.{} acc: {}".format(xval_fold, class_label, epoch_acc))
        save_metric(cur, model_name, xval_fold, class_label, epoch_acc)
    conn.commit()
    cur.close()
    conn.close()

def xval_iter(partition, n):
    for i in range(n):
        yield '{partition}::{fold}'.format(partition=partition, fold=chr(i+65))

def build_res50(fname_load=None):
    model = models.resnet50(pretrained=(fname_load is None))
    model.fc = torch.nn.Linear(in_features=2048, out_features=NUM_CLASS)
    if fname_load:
        print('Loading pretrained weights.')
        model.load_state_dict(torch.load(fname_load))
    return model.cuda()

def build_inception(fname_load=None):
    model = models.inception_v3(pretrained=(fname_load is None))
    model.fc = torch.nn.Linear(in_features=2048, out_features=NUM_CLASS)
    if fname_load:
        print('Loading pretrained weights.')
        model.load_state_dict(torch.load(fname_load))
    return model.cuda()

def build_densenet(fname_load=None):
    model = models.densenet121(pretrained=(fname_load is None))
    model.classifier = torch.nn.Linear(in_features=1024, out_features=NUM_CLASS)
    if fname_load:
        print('Loading pretrained weights.')
        model.load_state_dict(torch.load(fname_load))
    return model.cuda()

if __name__ == "__main__":
    # # Cross validation: Res50
    # for fold in xval_iter('5fold', 5):
    #     model = build_res50()
    #     train(model, EPOCHS, f'{OUTPUT}/res50.{fold}.pth', xval_fold=fold)

    # # Generate evaluation metrics
    # for fold in xval_iter('5fold', 5):
    #     model = build_res50(f'{OUTPUT}/res50.{fold}.pth')
    #     xval_evaluate(model, model_name='res50', xval_fold=fold)

    # # Cross validation: Inception
    # for fold in xval_iter('5fold', 5):
    #     model = build_inception()
    #     train(model, EPOCHS, f'{OUTPUT}/inception.{fold}.pth', xval_fold=fold, is_inception=True)

    # # Generate evaluation metrics
    # for fold in xval_iter('5fold', 5):
    #     model = build_inception(f'{OUTPUT}/inception.{fold}.pth')
    #     xval_evaluate(model, model_name='inception', xval_fold=fold)

    # Cross validation: DenseNet
    for fold in xval_iter('5fold', 5):
        model = build_densenet()
        train(model, EPOCHS, f'{OUTPUT}/densenet.{fold}.pth', xval_fold=fold)

    # Generate evaluation metrics
    for fold in xval_iter('5fold', 5):
        model = build_densenet(f'{OUTPUT}/densenet.{fold}.pth')
        xval_evaluate(model, model_name='densenet', xval_fold=fold)
