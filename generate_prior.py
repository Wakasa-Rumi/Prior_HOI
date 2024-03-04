import os
import torch
import warnings
import numpy as np
import pickle
import json
from PIL import Image
from tqdm.auto import tqdm

from dataloader.datasets import get_dataset
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from diffusers import DiffusionPipeline
import requests

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt

texts = {
    "024_bowl": "In clean white background,a red bowl. It is shaped like a bowl, with a circular shape and a red color."
}

paths = {
    "caption" : "data_preprocess/dexycb/caption",
    "mesh_image" : "data_preprocess/dexycb/mesh_image",
    "origin" : "data_preprocess/dexycb/origin",
    "pc_image" : "data_preprocess/dexycb/pc_image",
    "template" : "data_preprocess/dexycb/template",
    "json": "data_preprocess/dexycb",
}

classes = [
    "002_master_chef_can",
    "003_cracker_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "052_extra_large_clamp",
    "061_foam_brick",
    "004_sugar_box",
    "019_pitcher_base",
    "021_bleach_cleanser",
]

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def display_obj(verts, class_name, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=True, num=0):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts)
    # plt.savefig('visual/obj{}.png'.format(num))
    plt.savefig(paths["pc_image"] + '/{}.png'.format(class_name))

def generate_text():
    data_train, data_val, data_test = get_dataset(dataset_name="dexycb", model_name="GenMask")
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # instructBLIP
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = "Can yon describe the shape and color of object held by a hand in this image in detail within 60 words?"
    object_n = 0
    for step, (image_crop, object_name) in enumerate(data_loader_train):
        if object_name[0] in classes:
            continue
        object_n = object_n + 1

        image = image_crop[0].numpy()
        mask_pil = Image.fromarray(image.astype(np.uint8))
        mask_pil.save(paths["origin"] + "/{}.jpg".format(object_name[0]))

        inputs = processor(images=mask_pil, text=prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        generated_text = generated_text.replace('The object held by a hand in the image is ', '')
        generated_text = generated_text.replace('The object being held by a hand in the image is ', '')
        generated_text = generated_text.replace('The object that is held by a hand in the image is ', '')
        generated_text = "In clean white background," + generated_text.replace('hand', '')
        print("[{}] {}".format(object_name[0], generated_text))
        texts[object_name[0]] = generated_text

        json_str = json.dumps(texts, indent=4)
        with open(paths["json"] + "/text.json", "w", newline="\n") as f:
            f.write(json_str)

        if object_n > 5:
            break

def generate_synimage():
    pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/juggernaut-xl-v5")
    pipe.load_lora_weights("ehristoforu/dalle-3-xl")
    pipe = pipe.to("cuda")

    for key in texts.keys():
        class_name = key
        prompt = texts[class_name]
        image = pipe(prompt).images[0]
        image.save(paths["caption"] + '/{}.png'.format(class_name))

def point_e_process():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('creating base model...')
    base_name = 'base300M' # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, torch.device('cpu')))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', torch.device('cpu')))

    base_model.cuda()
    upsampler_model.cuda()

    for key in texts.keys():
        class_name = key
        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 3.0],
        )

        # Load an image to condition on.
        img = Image.open(paths["caption"] + '/{}.png'.format(class_name))

        # Produce a sample from the model.
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
            samples = x

        pc = sampler.output_to_point_clouds(samples)[0]
        # fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)), class_name=class_name)

        torch.save(pc.coords, paths["template"] + "/{}.pth".format(class_name))
        display_obj(pc.coords, class_name)

def generate_folder():
    for key in paths.keys():
        if not os.path.exists(paths[key]):
            os.mkdir(paths[key])

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print("start")
    generate_folder()
    # generate_text()
    generate_synimage()
    point_e_process()