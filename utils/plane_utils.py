# https://gist.github.com/tvercaut/d210ed103f64ebb24606eb032b38de0a

import torch
import numpy as np
import open3d as o3d


def fit_plane_batch_crossprod(points, normalize=False):
  # points: BxNx3 batched groups of N=3 3d points
  # return: Bx4 where for each input in the batch we get [a, b, c, d] such that
  # ax+by+cz+d = 0 is the equation of teh corresponding plane
  # No care is taken about colinear inputs
  num_batch = points.shape[0]
  num_points = points.shape[1]
  dim = points.shape[2]

  if dim != 3:
    raise ValueError(f"Only 3D points are supported, got dim={dim}")

  if num_points != 3:
    raise ValueError(f"Only groups of 3 points are supported, got num_points={num_points}")

  # We have to find the plane equation described by those 3 points
  # We find first 2 vectors that are part of this plane
  # V10 = pt1 - pt0
  # V20 = pt2 - pt0
  vec10 = points[:,1,:] - points[:,0,:]
  vec20 = points[:,2,:] - points[:,0,:]

  # Now we compute the cross product of vec10 and vec20 to get vecN which is normal to the plane
  vecN = torch.linalg.cross(vec10, vec20)
  
  # The plane equation will be vecN[0]*x + vecN[1]*y + vecN[0]*z = -d
  # We use any of the points to find k
  d = -torch.einsum('bd,bd->b', vecN, points[:,0,:].squeeze(dim=1))
  plane_params = torch.cat((vecN,d.unsqueeze(1)),1)

  # We can normalise for better interpretability but this would lead to issues
  # if the points are co-linear
  if normalize:
    plane_params /= torch.linalg.norm(plane_params,dim=1,keepdim=True)
  return plane_params

def plane_batch_dist(points, plane_params, signeddist=False):
  # plane_params: Bx4 B plane equations ax+by+cz+d = 0 given as [a, b, c, d] 
  # points: Nx3 N 3d points
  # return: BxN distances
  # No care is taken in case [a,b,c] has a zero norm
  num_batch = plane_params.shape[0]
  num_points = points.shape[0]
  dim = points.shape[1]

  if dim != 3:
    raise ValueError(f"Only 3D points are supported, got dim={dim}")

  # Distance from a point to a plane
  # https://mathworld.wolfram.com/Point-PlaneDistance.html
  # ax+by+cz+d / norm([a,b,c])
  denoms = torch.linalg.norm(plane_params[:,0:3],dim=1,keepdim=True)
  point_h = torch.cat((points, torch.ones((num_points,1),device=points.device)),1)
  dists = torch.einsum('bd,nd->bn', plane_params, point_h) / denoms
  if not signeddist:
    dists = torch.abs( dists )
  return dists

def fit_plane_batch_eigh(points):
  # points: BxNx3 batched groups of N 3d points
  # return: Bx4 where for each input in the batch we get [a, b, c, d] such that
  # ax+by+cz+d = 0 is the equation of teh corresponding plane
  # No care is taken about colinear inputs
  num_batch = points.shape[0]
  num_points = points.shape[1]
  dim = points.shape[2]

  if dim != 3:
    raise ValueError(f"Only 3D points are supported, got dim={dim}")

  points_h = torch.cat((points, torch.ones((num_batch,num_points,1),device=points.device)),2)
  smat = torch.matmul( torch.transpose(points_h,1,2), points_h )
  w, v = torch.linalg.eigh(smat)
  return v[:, :, 0]

def fit_plane_ransac(points, inlierthresh=0.05, batch=1000, numbatchiter=100, verbose=False):
  # points: Nx3 N 3d points
  # inlierthresh: Threshold distance from the plane which is considered inlier.
  # batch: Number of triplets considered in parallel during each batched RANSAC
  # iteration
  # numbatchiter: Number of maximum iteration which RANSAC will loop over.
  # return: [a, b, c, d] such that ax+by+cz+d = 0 is the equation of the
  # corresponding plane
  num_points = points.shape[0]
  dim = points.shape[1]

  if dim != 3:
    raise ValueError(f"Only 3D points are supported, got dim={dim}")

  best_num_inliers = 0
  best_inliers = None
  best_plane = None

  for iter in range(numbatchiter):
    if verbose:
      print(f"iteration: {iter}")
    # Get random triplets. Ideally, we would draw without replacement in each
    # triplet and also ensure we don't use duplicate triplets.
    randidx = torch.randint(0, num_points, (batch,3))
    triplets = points[randidx.flatten(),:].reshape(batch,3,3)

    # Get planes going through triplets
    planes = fit_plane_batch_crossprod(triplets)

    # Estimate quality of each plane
    dists = plane_batch_dist(points, planes)
    inliers = (dists < inlierthresh)
    num_inliers = torch.sum(inliers, dim=1)

    batch_best_idx = torch.argmax(num_inliers)
    batch_best_num_inliers = num_inliers[batch_best_idx]
    if verbose:
      print(f'  best batch plane: {planes[batch_best_idx,:] / torch.linalg.norm(planes[batch_best_idx,:])} - ({batch_best_num_inliers}/{num_points})')
    if batch_best_num_inliers > best_num_inliers:
      best_num_inliers = batch_best_num_inliers
      best_inliers = inliers[batch_best_idx,:]
      best_plane = planes[batch_best_idx,:]
      if verbose:
        print(f'best new plane: {best_plane / torch.linalg.norm(best_plane)} - ({best_num_inliers}/{num_points})')

  # Refine plane equation based on inliers
  inlier_points = points[best_inliers,:]
  best_plane = fit_plane_batch_eigh(inlier_points.unsqueeze(0))
  if verbose:
    print(f'refined_plane {best_plane}')
  return best_plane, best_inliers
     


def create_plane(normal, distance, size = 10.0, resolution = 10):

    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    xx, yy = np.meshgrid(x, y)

    a, b, c = normal
    d = distance
    if c == 0:  # Avoid division by zero
        raise ValueError("c cannot be zero for a valid plane.")
    zz = -(a * xx + b * yy + d) / c

    vertices = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx1 = i * resolution + j
            idx2 = i * resolution + (j + 1)
            idx3 = (i + 1) * resolution + j
            idx4 = (i + 1) * resolution + (j + 1)
            triangles.append([idx1, idx2, idx3])
            triangles.append([idx2, idx4, idx3])
            
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    plane_mesh.compute_vertex_normals()

    return plane_mesh