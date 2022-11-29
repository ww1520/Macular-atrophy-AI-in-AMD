import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader

net = UNet().cuda()
optimizer = torch.optim.Adam(net.parameters())
loss_func = nn.BCELoss()
data=SEGData()
dataloader = DataLoader(data, batch_size=16, shuffle=True,num_workers=0,drop_last=True)

module = r'C:\\Users\\lr3118\\Desktop\\data\module.pth'
img_save_path = r"C:\\Users\\lr3118\\Desktop\\data\tsave"

EPOCH=300

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
else:
    print("NO Params!")

if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)


for epoch in range(EPOCH):
    print('Round{}start'.format(epoch))
    net.train()
    for i,(img,label) in  enumerate(dataloader):
        img=img.cuda()
        label=label.cuda()
        img_out=net(img)
        loss=loss_func(img_out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    torch.save(net.state_dict(),module)
    
    
    print('Round{}end'.format(epoch))

