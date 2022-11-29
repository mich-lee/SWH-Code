################################################################################################################################
import numpy as np
import torch
from stl import mesh
################################################################################################################################

def createMeshFromGridsAndDepths(xGrid, yGrid, depthGrid, subsamplingMagnitude = 1):
	# Taking a subset of the data so the mesh does not get too big
	xGrid = xGrid[0::subsamplingMagnitude, 0::subsamplingMagnitude]
	yGrid = yGrid[0::subsamplingMagnitude, 0::subsamplingMagnitude]
	depthGrid = depthGrid[0::subsamplingMagnitude, 0::subsamplingMagnitude]


	nHeight = xGrid.shape[0]
	nWidth = xGrid.shape[1]

	indsLinear = torch.tensor(range(nHeight*nWidth))

	# gridInds = indsLinear.view(nHeight, nWidth) # Note that this will point to the same memory as linearInds
	# xGridInds, yGridInds = generateGrid(res=(nHeight,nWidth), deltaX=1, deltaY=1, centerGrids=False)
	
	indsLinearX = torch.floor(indsLinear / nWidth)
	indsLinearY = indsLinear % nWidth

	vertices = torch.zeros(nHeight, nWidth, 3)
	vertices[:,:,0] = xGrid
	vertices[:,:,1] = yGrid
	vertices[:,:,2] = depthGrid

	vertices = vertices.view(nHeight * nWidth, 3)


	vInds0 = (nWidth * indsLinearX) + indsLinearY

	# Upper triangles
	selectedIndsA = (indsLinearX < (nHeight - 1)) & (indsLinearY < (nWidth - 1))
	vIndsA1 = (nWidth * (indsLinearX + 1)) + indsLinearY
	vIndsA2 = (nWidth * indsLinearX) + (indsLinearY + 1)
	vIndsA0 = vInds0[selectedIndsA]
	vIndsA1 = vIndsA1[selectedIndsA]
	vIndsA2 = vIndsA2[selectedIndsA]

	# Lower triangles
	selectedIndsB = (indsLinearX > 0) & (indsLinearY > 0)
	vIndsB1 = (nWidth * (indsLinearX - 1)) + indsLinearY
	vIndsB2 = (nWidth * indsLinearX) + (indsLinearY - 1)
	vIndsB0 = vInds0[selectedIndsB]
	vIndsB1 = vIndsB1[selectedIndsB]
	vIndsB2 = vIndsB2[selectedIndsB]

	faces = torch.cat(
						(
							torch.cat((vIndsA0[:,None], vIndsA1[:,None], vIndsA2[:,None]), dim=1),
							torch.cat((vIndsB0[:,None], vIndsB1[:,None], vIndsB2[:,None]), dim=1)
						),
						dim=0
					)

	depthMesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
	depthMesh.vectors = vertices[faces.to(torch.long),:]

	# Used for verification:
		# for i in range(height):
		# 	print(i)
		# 	for j in range(width):
		# 		vertA = vertices[(i*width) + j]
		# 		vertB = torch.tensor([xGrid[i,j], yGrid[i,j], depthGrid[i,j]])
		# 		# print(vertA, vertB, (vertA - vertB).abs().sum() == 0)
		# 		if ((vertA - vertB).abs().sum() != 0):
		# 			raise Exception("asdf")

	return depthMesh