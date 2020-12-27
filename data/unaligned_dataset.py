import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch

class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # get the images path
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        
        # put all paths of images in a list
        self.A_paths = make_dataset(self.dir_A) 
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform = get_transform(opt)

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)

        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

    def rgb2yCbCr(self, input_im):
        input_im = input_im.clone()
        im_flat = input_im.view(-1, 3)
        mat = torch.tensor([[0.257, -0.148, 0.439],
                            [0.564, -0.291, -0.368],
                            [0.098, 0.439, -0.071]])
        bias = torch.tensor([16.0/255.0, 128.0/255.0, 128.0/255.0])
        # for i in range(batch):
            # print(im_flat[i].shape)
        temp = im_flat.mm(mat) + bias
        # print(input_im.size())
        out = temp.view(3, input_im.shape[1], input_im.shape[2])
        # out = out.permute(2, 0, 1)
        out_y = out[0,:,:].view(1, input_im.shape[1], input_im.shape[2])
        # out_y = np.transpose(out_y, axes=[2, 0, 1])
        out_uv = out[1:,:,:].view(2, input_im.shape[1], input_im.shape[2])
        # out_uv = np.transpose(out_uv, axes=[2, 0, 1])
        return out_y, out_uv
    
    def yCbCr2rgb(self, input_im):

        # out = color.ycbcr_to_rgb(input_im)
        im_flat = input_im.view(-1, 3).cuda()
        mat = torch.tensor([[1.164, 1.164, 1.164],
                        [0, -0.392, 2.017],
                        [1.596, -0.813, 0]]).cuda()
        bias = torch.tensor([-16.0/255.0, -128.0/255.0, -128.0/255.0]).cuda()
        temp = (im_flat + bias).mm(mat)
        out = temp.view(-1, 3, list(input_im.size())[2], list(input_im.size())[3])
        return out
    
    def gamma(self, input_im):
        input_y, input_uv = self.rgb2yCbCr(input_im)
        input_y_gamma = torch.pow(input_y, 0.4)
        input_yuv = torch.cat([input_y_gamma, input_uv], dim=1)
        input_rgb = self.yCbCr2rgb(input_yuv)
        return input_rgb
