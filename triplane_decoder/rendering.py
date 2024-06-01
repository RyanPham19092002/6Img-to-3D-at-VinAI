import torch
import triplane_decoder.ray_samplers as ray_sampler
from triplane_decoder.losses import distortion_loss
from triplane_decoder.pif import PIF
from triplane_decoder.decoder import TriplaneDecoder
from visualize import visualize_3d_points_and_directions
from visualize import visualize_random_rays
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np

def extract_planes_info(triplane_decoder, points):
    # Assume that points is a tensor of shape [N, 3] containing the coordinates of the points
    
    # Convert points to numpy array
    points_np = points.detach().cpu().numpy()
    
    # Pass the points through the decoder_net to obtain plane coefficients
    with torch.no_grad():
        plane_coeffs = triplane_decoder.decoder_net(points)
    
    # Convert plane coefficients to numpy array
    plane_coeffs_np = plane_coeffs.cpu().numpy()
    
    # Assuming plane_coeffs_np has shape [N, 4], where each row contains the coefficients (a, b, c, d)
    # of the plane equation ax + by + cz + d = 0
    
    return points_np, plane_coeffs_np

def visualize_planes_info(points, plane_coeffs, filename="Triplane decoder"):
    # Assuming points and plane_coeffs are numpy arrays
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label='Points')

    # Plot planes
    for coeff in plane_coeffs:
        a, b, c, d = coeff
        xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        zz = (-a * xx - b * yy - d) / c
        ax.plot_surface(xx, yy, zz, alpha=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Visualization of Points and Planes')
    plt.savefig(filename, format='png')
    plt.show()

def plot_points_3d(x, name):
    x = x.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    indices = np.random.choice(len(x), len(x), replace=False)
    selec_x = x[indices]
    ax.scatter(selec_x[:,0], selec_x[:,1], selec_x[:,2], c='b', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(name)



def plot_ray_directions(x, ray_directions, name):
    """
    Plot ray directions in 3D space.

    Args:
        ray_directions (torch.Tensor): Ray directions to plot, shape (num_rays * num_samples, 3).
        name (str): The filename to save the plot.
    """
    # Convert tensor to numpy array for plotting
    ray_directions = ray_directions.detach().cpu().numpy()
    x = x.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Đang vẽ")
    # Set to keep track of plotted directions
    print("len x, len ray", len(x), len(ray_directions))

    for origin, direction in zip(x, ray_directions):
        
        # Vẽ ray direction là màu đỏ
        ax.quiver(origin[0], origin[1], origin[2], 
                direction[0], direction[1], direction[2], 
                length=1.0, normalize=True, color='r', alpha=0.5)
        ax.quiver(origin[0], origin[1], origin[2], 
                0, 0, 0,  # Hướng không quan trọng ở đây vì chỉ là một điểm
                length=1.0, normalize=True, color='b', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig(name)
    print("Đã vẽ xong")
    plt.show()




def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1) #rays, #samples, 1
    return torch.cat((torch.ones((accumulated_transmittance.size(0), 1, 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=1)


def ray_aabb_intersection(ray_origins, ray_directions, aabb_min, aabb_max):
    tmin = (aabb_min - ray_origins) / ray_directions
    tmax = (aabb_max - ray_origins) / ray_directions
    
    t1 = torch.min(tmin, tmax)
    t2 = torch.max(tmin, tmax)
    
    t_enter = torch.max(t1,dim=1).values
    t_exit = torch.min(t2,dim=1).values
    
    return t_enter, t_exit

def render_rays(nerf_model:TriplaneDecoder, ray_origins, ray_directions, config, triplane=None, pif: PIF = None,  training=True, only_coarse=False, **kwargs):
    device = ray_origins.device
    # points, plane_coeffs = extract_planes_info(nerf_model, ray_origins)
    # visualize_planes_info(points, plane_coeffs)
    # Gọi hàm để trực quan hóa triplane_decoder
    #visualize_triplane_decoder(TriplaneDecoder)
    #print("pif---------", pif)
    
    uniform_sampler = ray_sampler.UniformSampler(num_samples=config.decoder["nb_bins"], train_stratified=config.decoder["train_stratified"])
    pdf_sampler = ray_sampler.PDFSampler(num_samples=config.decoder["nb_bins"], train_stratified=config.decoder["train_stratified"], include_original=False)

    uniform_sampler.training = training
    pdf_sampler.training = training
    #print("hf-----------------", config.decoder["hf"])
    ray_bundle = ray_sampler.RayBundle(
        origins=ray_origins,
        directions=ray_directions,
        nears=torch.ones((ray_origins.size(0),1), device=device) * config.decoder["hn"],
        fars=torch.ones((ray_origins.size(0),1), device=device)  * config.decoder["hf"],
    )
    #visualize_random_rays(ray_origins.detach().cpu().numpy(), ray_directions.detach().cpu().numpy(), num_rays=9216, ray_length=1.0, name = 'visualize_ray_start_to_end.png')
    # Coarse sampling
    samples_coarse = uniform_sampler.generate_ray_samples(ray_bundle)
    #print("---------- sample coarse1", samples_coarse.starts.shape, samples_coarse.ends.shape)
    midpoints = (samples_coarse.starts + samples_coarse.ends) / 2                 #rays, #samples, 1
    #print("midpoints-----------------------", midpoints.shape)
    x = samples_coarse.origins + samples_coarse.directions.squeeze(2) * midpoints #rays, #samples, 3
    #plot_points_3d(x, "samples_coarse.png")
    # print("x before-----------------------", x.reshape(-1,3))
    viewing_directions = ray_directions.expand(x.size(1), -1, 3).permute(1,0,2)   #rays, #samples, 3

    # print("viewing_directions before -----------------------", viewing_directions.reshape(-1,3))
    # Example usage with x and viewing_directions from coarse sampling
#    visualize_points_and_directions_2d(x, viewing_directions, 'coarse sampling.png', len(x))
    colors, densities = nerf_model(x.reshape(-1,3), viewing_directions.reshape(-1,3), pif=pif)
    colors_coarse = colors.reshape_as(x)               #rays, #samples, 3 
    densities_coarse = densities.reshape_as(midpoints) #rays, #samples, 1
    #print("densities_coarse shape------", densities_coarse.shape)
    #print("colors_coarse-----------------------", colors_coarse.shape)
    #print("densities_coarse-----------------------", densities_coarse.shape)
    weights = samples_coarse.get_weights(densities_coarse) #rays, #samples, 1
    #print("weights-----------------------", weights.shape)
    #exit(0)
    if only_coarse:
        colors = volume_rendering(samples_coarse.deltas,
                        colors_coarse, 
                        densities_coarse, 
                        config.decoder.white_background)

        dist_loss = distortion_loss(weights, samples_coarse)

        depth = get_depth(weights, midpoints)
        return colors, dist_loss, depth


    # Fine sampling
    samples_fine = pdf_sampler.generate_ray_samples(ray_bundle, samples_coarse, weights,  config.decoder["nb_bins"])

    midpoints = (samples_fine.starts + samples_fine.ends) / 2 #rays, #samples, 1
    x = samples_coarse.origins + samples_coarse.directions.squeeze(2) * midpoints #rays, #samples, 3   
    #plot_points_3d(x.reshape(-1,3), "samples_fine.png")     
    # print("x after-----------------------", x.reshape(-1,3))
    #visualize_points_and_directions_2d(x, ray_directions, 'fine sampling.png', len(x))
    #print("x shape------------------------------------------", x.reshape(-1, 3).shape)
    # #visualize
    # print("ray_directions shape------------------------------------------", ray_directions.reshape(-1, 3).shape)
    # visualize_3d_points_and_directions(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    #print("x.reshape(-1,3)", x.reshape(-1,3).shape)
    #print("ray_directions.reshape(-1,3)", ray_directions.reshape(-1,3).shape)
    #visualize_random_rays(x.reshape(-1,3).detach().cpu().numpy(), ray_directions.reshape(-1, 3).repeat(64, 1).detach().cpu().numpy(), num_rays=len(x), ray_length=60.0, name = 'ray_directions.png')
    #print("ray_directions.reshape(-1, 3)", ray_directions.reshape(-1, 3))
    colors, densities = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3), pif=pif)   #origin ray_directions
    colors_fine = colors.reshape_as(x)               #rays, #samples_per_ray, 3
    densities_fine = densities.reshape_as(midpoints) #rays, #samples_per_ray
    # print("colors_fine------", colors_fine)
    #print("samples_fine.deltas------", samples_fine.deltas)
    
    colors = volume_rendering(samples_fine.deltas,
                            colors_fine, 
                            densities_fine, 
                            config.decoder.white_background)
    # print("colors------", colors)
    weights = samples_fine.get_weights(densities_fine) #rays, #samples, 1
    # print("weights------", weights)
    dist_loss = distortion_loss(weights, samples_fine)

    depth = get_depth(weights, midpoints)

    return colors, dist_loss, depth

def volume_rendering(deltas: torch.Tensor, colors: torch.Tensor, sigma: torch.Tensor, white_background: bool) -> torch.Tensor:
    alpha = 1 - torch.exp(-sigma * deltas)  #rays, #samples, 1
    weights = compute_accumulated_transmittance(1 - alpha) * alpha #rays, #samples, 1
    colors = (weights * colors).sum(dim=1)  #rays, 3
    # print("alpha-----------", alpha)
    # print("weights-----------", weights)
    # print("colors-----------", colors)
    if white_background:
        weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
        colors = colors + 1 - weight_sum.unsqueeze(-1)  #samples, 3

    return colors #rays, 3


def get_depth(weights, steps):
    """
    https://docs.nerf.studio/_modules/nerfstudio/model_components/renderers.html#DepthRenderer
    """
    cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [..., num_samples]
    split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5  # [..., 1]
    median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
    median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)  # [..., 1]
    median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)  # [..., 1]
    return median_depth
