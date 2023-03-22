import torch
from gan_training import utils
from gan_training.metrics import inception_score
from gan_training.metrics.fid_score import calculate_fid_given_images
from gan_metrics.kid_score import *
from gan_metrics.precision_recall import *
import numpy as np
import lpips
import shutil
import random


class Evaluator(object):
    def __init__(self, args, generator, batch_size=64,
                 inception_nsamples=5000, device=None, fid_real_samples=None,
                 fid_sample_size=5000, compute_pr=False):
        self.args = args
        self.generator = generator
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device
        self.sample_size = fid_sample_size
        if fid_real_samples is not None:
            np.save(os.path.join(args.output_path, "real_imgs.npy"), fid_real_samples.numpy())
            self.sample_size = fid_sample_size

        if compute_pr:
            # to compute precision-recall
            self.ipr = IPR(self.batch_size, k=3, num_samples=self.sample_size)

    def compute_inception_score(self, fid=True, kid=False, pr=False):
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            sample_z = torch.randn(self.args.n_sample_store, self.args.latent).cuda()
            try:
                samples, _ = self.generator([sample_z.data])
            except:
                samples = self.generator([sample_z.data])
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        # for different inception scores
        score={} # init score dict
        real_imgs = np.load(os.path.join(self.args.output_path, "real_imgs.npy"))
        fake_imgs = np.array(imgs[:self.sample_size])

        # fid
        if fid:
            fid = calculate_fid_given_images(real_imgs, fake_imgs, self.batch_size, cuda=True)
            score['fid'] = fid

        # kid
        if kid:
            kid = calculate_kid_given_images(real_imgs[:2000], fake_imgs[:2000], self.batch_size, cuda=True)
            score['kid'] = kid[0]

        # precision-recall
        if pr:
            # real
            self.ipr.compute_manifold_ref(real_imgs)
            # fake
            precision, recall = self.ipr.precision_and_recall(fake_imgs)
            score['precision'] = precision
            score['recall'] = recall
        return score


    def create_samples(self, z, y=None):
        self.generator.eval()
        batch_size = z.size(0)
        # Parse y
        if y is None:
            y = self.ydist.sample((batch_size,))
        elif isinstance(y, int):
            y = torch.full((batch_size,), y,
                           device=self.device, dtype=torch.int64)
        # Sample x
        with torch.no_grad():
            x = self.generator(z, y)
        return x

    def compute_intra_lpips(self, args):
        del_assigned_images(args)
        self.generator.eval()
        imgs = []
        while(len(imgs) < 1000):
            sample_z = torch.randn(self.args.n_sample_store, self.args.latent).cuda()
            samples, _ = self.generator([sample_z.data])
            imgs.extend(samples)

        fake_imgs = imgs[:self.sample_size] # tensor imgs to be saved

        # step 1. save each of samples to args.intra_lpips_path
        for (idx, sample) in enumerate(fake_imgs):
            utils.save_images(sample, f"%s/{str(idx).zfill(6)}.png" % (args.intra_lpips_path))

        # step 2. assign generated images to cluster centers
        assign_fake_images_to_cluster_center(args)

        # step 3. compute intra-lpips
        intra_lpips_dist = intra_cluster_dist(args)

        # step 4. delete abundant images of this checkpoint
        del_assigned_images(args)

        return intra_lpips_dist


def assign_fake_images_to_cluster_center(args):
    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net='vgg').cuda()
        preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        center_path = os.path.join(f"../cluster_centers", args.data_path, args.method)
        files_list_samples = os.listdir(args.intra_lpips_path)

        # Step 1. compute lpips between sample and center
        for i in tqdm(range(len(files_list_samples))):  # all generated samples
            dists = []
            for k in range(10):  # cluster center
                cluster_center = os.path.join(center_path, "c%d" % (k), "center.png")
                input1_path = os.path.join(args.intra_lpips_path, files_list_samples[i])
                input2_path = os.path.join(cluster_center)

                input_image1 = Image.open(input1_path).convert('RGB')
                input_image2 = Image.open(input2_path).convert('RGB')

                input_tensor1 = preprocess(input_image1)
                input_tensor2 = preprocess(input_image2)

                input_tensor1 = input_tensor1.cuda()
                input_tensor2 = input_tensor2.cuda()

                dist = lpips_fn(input_tensor1, input_tensor2)
                dists.append(dist.cpu())
            dists = np.array(dists)

            # Step 2. Move images close to the best matched cluster
            idx = np.argmin(dists)
            shutil.move(input1_path, os.path.join(center_path, "c%d" % (idx)))


def intra_cluster_dist(args):
    with torch.no_grad():
        lpips_fn = lpips.LPIPS(net='vgg').cuda()
        preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        cluster_size = 50
        base_path = os.path.join(f"../cluster_centers", args.data_path, args.method)
        avg_dist = torch.zeros([10, ])  # placeholder for intra-cluster lpips
        for k in range(10):
            curr_path = os.path.join(base_path, "c%d" % (k))
            files_list = os.listdir(curr_path)
            files_list.remove('center.png')

            random.shuffle(files_list)
            files_list = files_list[:cluster_size]
            dists = []
            for i in range(len(files_list)):
                for j in range(i+1, len(files_list)):
                    input1_path = os.path.join(curr_path, files_list[i])
                    input2_path = os.path.join(curr_path, files_list[j])
                    # input2_path = os.path.join(curr_path, 'center.png')
                    input_image1 = Image.open(input1_path)
                    input_image2 = Image.open(input2_path)

                    input_tensor1 = preprocess(input_image1)
                    input_tensor2 = preprocess(input_image2)

                    input_tensor1 = input_tensor1.cuda()
                    input_tensor2 = input_tensor2.cuda()

                    dist = lpips_fn(input_tensor1, input_tensor2)

                    dists.append(dist.cpu())
            dists = torch.tensor(dists)
            print ("Cluster %d:  Avg. pairwise LPIPS dist: %f/%f" %
                   (k, dists.mean(), dists.std()))
            avg_dist[k] = dists.mean()

        # print ("Final avg. %f/%f" % (avg_dist[~torch.isnan(avg_dist)].mean(), avg_dist[~torch.isnan(avg_dist)].std()))
        return avg_dist[~torch.isnan(avg_dist)].mean()


def del_assigned_images(args):

    # remove images around cluster center
    base_path = os.path.join(f"../cluster_centers", args.data_path, args.method)
    for k in range(10):
        curr_path = os.path.join(base_path, "c%d" % (k))
        files_list = os.listdir(curr_path)
        files_list.remove('center.png')

        for i in range(len(files_list)):
            img = os.path.join(curr_path, files_list[i])
            if os.path.exists(img):
                os.remove(img)
            else:
                print("The file does not exist")
    print ("assigned images deleted" )

    # clear abundant generated images (if any left)
    files_list = os.listdir(args.intra_lpips_path)

    for i in range(len(files_list)):
        img = os.path.join(args.intra_lpips_path, files_list[i])
        if os.path.exists(img):
            os.remove(img)
        else:
            print("The file does not exist")
    print ("generated images deleted" )