from glob import glob
from torchvision.datasets import CIFAR10
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm
import csv


def get_ancer_sigma(sigma_folder_path: str, i: int):
    theta_i = torch.relu(
        torch.load(sigma_folder_path+'/sigma_test_'+str(i)+'.pth', map_location=torch.device('cpu'))
    )

    return theta_i


def main(args):

    g = args.results_file_path

    f = open(str(g), "r")
    count = 0
    index, label, prediction, min_radius, max_radius, correct, proxy_radius = [], [], [], [], [], [], []
    with open(str(g)) as f:
        reader = csv.DictReader(f, delimiter="\t")
        # second_counter += 1
        for row_i, row in enumerate(reader):
            if count > 0:
                # if second_counter%5 != 0:
                #     continue
                min_radius.append(float(row["radius"]))

                correct.append(int(row["correct"]))
                prediction.append(int(row["predict"]))
                label.append(int(row["label"]))
                index.append(int(row["idx"]))
                proxy_radius.append(float(row["proxyvol"]))

                # obtain the maximum l_p radius by using the gap computed from the min one
                optim_sigmas = get_ancer_sigma(args.optimized_sigmas, row_i)
                max_radius.append(min_radius[-1]*(optim_sigmas.max()/optim_sigmas.min()))
                
            else:
                count += 1
                # second_counter = 0

            # for debugging purposes only
            if row_i > 10:
                break

    print("loaded all data...")

    dataset = CIFAR10(root='./train/datasets', train=False, download=True, transform=ToTensor())

    saved_images, saved_predictions, saved_min_radii, saved_max_radii, saved_proxy_rad, keep_original_sigmas = [], [], [], [], [], []
    # anything_detected = False
    for i in tqdm(range(len(min_radius))):
        
        idx, pred, min_rad, max_rad, proxy_rad = index[i], prediction[i], min_radius[i], max_radius[i], proxy_radius[i]
        img, _ = dataset[idx]

        print(f"----------- New point {idx} ----------")

        # a variable indicating whether we should get the original sigmas or consider it a ball
        keep_sigma = True
        
        if len(saved_images) != 0:
            # Get the differences
            diff = torch.norm(img.reshape(1, -1) - torch.stack(saved_images).reshape(len(saved_max_radii), -1), dim=1)
            
            where_max_overlap = diff < (torch.tensor(saved_max_radii) + max_rad)

            # Check whether this image is with overlap with any other instances
            if where_max_overlap.any():
                print("- Maximums overlap")

                preds_max_overlap = torch.tensor(saved_predictions)[where_max_overlap]
                where_max_overlap_diff_class = preds_max_overlap != pred

                diff_where_max_overlap = diff[where_max_overlap]
                saved_min_radii_where_max_overlap = torch.tensor(saved_min_radii)[where_max_overlap]

                if where_max_overlap_diff_class.any():
                    print("-- Maximums between different predictions overlap! Adjust based on box...")

                    diff_where_max_overlap_diff_class = diff_where_max_overlap[where_max_overlap_diff_class]
                    saved_min_radii_where_max_overlap_diff_class = saved_min_radii_where_max_overlap[where_max_overlap_diff_class]
                    preds_max_overlap_diff_class = preds_max_overlap[where_max_overlap_diff_class]

                    # load the sigmas of the new point we're inferrencing on
                    # and build the B matrix as per the paper
                    B_sigmas = get_ancer_sigma(args.optimized_sigmas, i)
                    B_sigmas = B_sigmas / (min_rad/B_sigmas.min())**2
                    b = img
                    keep_sigma = False

                    # get indices where different classes overlap
                    overall_overlap_indices = torch.where(where_max_overlap)[0]
                    diff_class_overlap_indices = torch.where(where_max_overlap_diff_class)[0]

                    # for each of them, get the maximum box; overall box will be the minimum of these ones
                    for overall_idx, diff_idx in zip(overall_overlap_indices, diff_class_overlap_indices):
                        A_min_radius = saved_min_radii[overall_idx]

                        # if it's the original sigmas, load them; otherwise
                        # just take A_sigmas to be a ball of radius saved_min_radii
                        if keep_original_sigmas[overall_idx]:
                            A_sigmas = get_ancer_sigma(args.optimized_sigmas, diff_idx.item())
                            A_gap = A_min_radius/A_sigmas.min()
                            A_sigmas = 1/torch.sqrt(A_sigmas) / A_gap**2
                        else:
                            A_gap = 1
                            A_sigmas = torch.ones_like(B_sigmas) / A_min_radius**2

                        a, _ = dataset[overall_idx]

                        v_candidates = torch.maximum(1/(b - a - A_sigmas), 1/(a - b - A_sigmas))
                        if not torch.all(v_candidates > 0):
                            print("--- Failed box adjustment...")
                            if torch.linalg.norm(A_sigmas.flatten() * (b - a).flatten()) <= 1:
                                print("--- Failed for point inside the ellipsoid")
                            else:
                                print("--- Failed for point outside the ellipsoid but inside box")

                            # box adjustment did not work, will abstain
                            pred = -1
                            min_rad = 0
                            max_rad = 0
                            proxy_rad = 0
                            keep_sigma = False
                            break

                        # adjustment works, now make sure it's a subset of B_sigmas
                        v_candidates = torch.maximum(v_candidates, B_sigmas)
                        min_rad = torch.min(min_rad, 1/torch.max(v_candidates))
                        max_rad = min_rad

        saved_images.append(img)
        saved_predictions.append(pred)
        saved_min_radii.append(min_rad)
        saved_max_radii.append(max_rad)
        saved_proxy_rad.append(proxy_rad)
        keep_original_sigmas.append(keep_sigma)

        print("Done with point")

    import pdb
    pdb.set_trace()

    print("You are Done!, --------> Saving results")
    
    f = open(args.outfile, 'w')
    print("idx\tpredict\tradius-min\tradius-max\tproxyvol\tcorrect")
    for i in range(len(index)):
        print("{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}".format(
            index[i], saved_predictions[i], saved_min_radii[i], saved_max_radii[i], saved_proxy_rad[i], correct[i]), file=f, flush=True)

    print("You are officially done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--id', type=int, default=128, help="id of the path")
    parser.add_argument(
        "--results-file-path", type=str,
        help="path to the certification results file"
    )
    parser.add_argument(
        "--optimized-sigmas", type=str,
        help="path to the ANCER optimized sigmas folder"
    )
    parser.add_argument(
        "--new_results_path", type=str,
        help="path to the ANCER optimized sigmas folder"
    )

    args = parser.parse_args()
    main(args)
