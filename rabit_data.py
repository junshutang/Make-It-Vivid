import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import jsonlines
import glob
from torchvision import transforms
import random

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

def load_obj_mesh(mesh_file, uv_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        # elif values[0] == 'vn':
        #     vn = list(map(float, values[1:4]))
        #     norm_data.append(vn)
        # elif values[0] == 'vt':
        #     vt = list(map(float, values[1:3]))
        #     uv_data.append(vt)

    with open(uv_file, "r") as temp_file:
        for line in temp_file.readlines():
            line = line.replace('\n', '')
            if line.startswith('#'):
                continue
            values = line.split()
            if len(values) == 0:
                continue
            elif values[0] == "vt":
                vt = list(map(float, values[1:3]))
                uv_data.append(vt)
            elif values[0] == 'f':
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                    face_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                    face_data.append(f)
                # tri mesh
                else:
                    f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                    face_data.append(f)
                
                # deal with texture
                if len(values[1].split('/')) >= 2:
                    # quad mesh
                    if len(values) > 4:
                        f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                        face_uv_data.append(f)
                        f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                        face_uv_data.append(f)
                    # tri mesh
                    elif len(values[1].split('/')[1]) != 0:
                        f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                        face_uv_data.append(f)
                # deal with normal
                if len(values[1].split('/')) == 3:
                    # quad mesh
                    if len(values) > 4:
                        f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                        face_norm_data.append(f)
                        f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                        face_norm_data.append(f)
                    # tri mesh
                    elif len(values[1].split('/')[2]) != 0:
                        f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                        face_norm_data.append(f)
    
    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces

class BiCarDataset(Dataset):
    def __init__(self, dataset_folder, device, input_size=1024):
        self.dataset_folder = os.path.join(dataset_folder, "3DBiCar")
        # self.data_index_list = os.listdir(self.dataset_folder)
        self.input_size = input_size
        self.uv_path = os.path.join(dataset_folder, "rabit_data/UV/tri.obj")
        self.device = device
        self.meta_file = os.path.join(dataset_folder, "3DBiCar/uv-train/metadata.jsonl")
        self.meta_list = []
        with jsonlines.open(self.meta_file) as f:
            for line in f:
                self.meta_list.append(line)
        print(len(self.meta_list))
        print(self.meta_list[0])
        # self.meta_list = self.meta_list[:100]
        
    @staticmethod
    def calculate_face_normals(vertices: torch.Tensor, faces: torch.Tensor):
        """
        calculate per face normals from vertices and faces
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        e0 = v1 - v0
        e1 = v2 - v0
        n = torch.cross(e0, e1, dim=-1)
        twice_area = torch.norm(n, dim=-1)
        n = n / twice_area[:, None]
        return n, twice_area / 2
        
    def __getitem__(self, index):
        
        instance_name = self.meta_list[index]['file_name']
        instance_index = instance_name.split('.png')[0]
        caption = self.meta_list[index]['text']
        instance_folder = os.path.join(self.dataset_folder,instance_index)
        
        # beta = np.load(os.path.join(instance_folder,'params','beta.npy'))[:100]
        # theta = np.load(os.path.join(instance_folder,'params','pose.npy')).reshape(3,24)
        
        #mesh: Here we only read points and uvmap of body only.
        t_verts, t_faces, t_uvs, t_in_uv_faces = load_obj_mesh(os.path.join(instance_folder,'tpose','m.obj'), self.uv_path, with_texture=True)
        tbody_uv =  torch.Tensor(np.array(Image.open(os.path.join(self.dataset_folder, "uv-train", instance_name)).resize(
                (2048, 2048)))).permute(2, 0, 1) / 255.0
        # tbody_uv =  torch.Tensor(np.array(Image.open(os.path.join(instance_folder,'tpose','m.BMP')).resize(
        #         (2048, 2048)))).permute(2, 0, 1) / 255.0
        
        # tbody_uv = Image.open(os.path.join(instance_folder,'tpose','m.png'))
        vertices = torch.FloatTensor(t_verts)
        faces = torch.tensor(t_faces)
        normals, face_area = self.calculate_face_normals(vertices, faces)
        ft = torch.tensor(t_in_uv_faces)
        vt = torch.FloatTensor(t_uvs)
        
        # caption_path = 
        
        # return {'ref_image':image,
        return {'index':instance_index,
                'vertices':vertices,
                'faces':faces,
                'texture':tbody_uv,
                'normals':normals,
                'ft':ft, 
                'vt':vt,
                'text':caption
                } 
        
    def __len__(self):
        return len(self.meta_list)

    
class RenderDataset(Dataset):
    def __init__(self, dataset_folder, device, input_size=512):

        self.dataset_folder = dataset_folder
        self.dir_list = os.listdir(dataset_folder)
        self.device = device
        self.meta_file = os.path.join("metadata.jsonl")
        self.meta_list = {}
        with jsonlines.open(self.meta_file) as f:
            for line in f:
                self.meta_list.update(line)
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
                                         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) 
        self.blank = " "

    def __getitem__(self, index):
        
        dir_name = self.dir_list[index]
        random_idx = random.randint(0, 11)
        # random_idx_2 = random.randint(0, 4)
        # data_name = os.path.join(self.dataset_folder, dir_name, f"{random_idx}_{random_idx_2}_img.png")
        data_name = os.path.join(self.dataset_folder, dir_name, f"{random_idx:03d}.png")
        caption = self.meta_list[dir_name]
        image = Image.open(data_name)
        image = image.convert("RGB")
        image = self.trans(image)
        
        return {'pixel_values': image,
                'text': caption
                } 
        
    def __len__(self):
        return len(self.dir_list)

if __name__ == '__main__':
    dataset = BiCarDataset(dataset_folder="3DBiCar", device='cuda')
    batch_size = 2
    dataset.__getitem__(1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        for item in batch:
            try:
                print(item, batch[item].shape)
            except:
                print(item, batch[item])