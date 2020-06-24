import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import os
from termcolor import cprint
from tqdm import tqdm
from torchvision.transforms import ToTensor,Compose
from model import MnistNet
def train(cfg,net,train_dl,eval_dl):
    Loss=torch.nn.CrossEntropyLoss()
    optim=torch.optim.Adam(net.parameters(),lr=cfg["lr"])


    best_acc=0
    writer=SummaryWriter(log_dir=cfg["log_dir"])
    writer.add_graph(net,next(iter(train_dl))[0].cuda())
    if not os.path.isdir(cfg["checkpoints"]):
        os.makedirs(cfg["checkpoints"])
    for epoch in range(cfg["epochs"]):
        acc_cnt=0.
        train_sample_cnt=0
        eval_sample_cnt=0
        for batch_img,batch_target in tqdm(train_dl):
            batch_img=batch_img.cuda()
            batch_target=batch_target.cuda()
            optim.zero_grad()
            batch_predict=net(batch_img)
            loss=Loss(batch_predict,batch_target)
            loss.backward()
            optim.step()
            acc_cnt+=torch.sum(torch.argmax(batch_predict,dim=1)==batch_target)
            train_sample_cnt+=len(batch_target)
        train_acc=acc_cnt/float(train_sample_cnt)
        writer.add_scalar("train_acc",train_acc,epoch)
        cprint(f"epoch {epoch}, train_acc {train_acc}","green")
        acc_cnt=0
        for batch_img,batch_target in eval_dl:
            eval_sample_cnt+=len(batch_target)
            batch_img=batch_img.cuda()
            batch_target=batch_target.cuda()
            batch_predict=net(batch_img)
            acc_cnt+=torch.sum(torch.argmax(batch_predict,dim=1)==batch_target)
        eval_acc=acc_cnt/float(eval_sample_cnt)
        writer.add_scalar("eval_acc",eval_acc,epoch)
        cprint(f"epoch {epoch}, eval_acc {eval_acc}","green")
        if eval_acc>best_acc:
            best_acc=eval_acc
            torch.save(net.state_dict(),os.path.join(cfg["checkpoints"],f"{epoch}_{best_acc:.5f}.pt"))

if __name__ == "__main__":
    data_dir="data"
    with open("config.json","r") as f:
        cfg=json.load(f)
    transform=Compose([ToTensor()])
    
    train_ds=torchvision.datasets.MNIST(data_dir,train=True,download=True,transform=transform)
    eval_ds=torchvision.datasets.MNIST(data_dir,train=False,download=True,transform=transform)
    train_dl=DataLoader(train_ds,batch_size=cfg["train_batch"],drop_last=True,shuffle=True)
    eval_dl=DataLoader(eval_ds,batch_size=cfg["eval_batch"],drop_last=True,shuffle=False)
    net=MnistNet()
    net.cuda()
    train(cfg,net,train_dl,eval_dl)



