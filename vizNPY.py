import numpy as np
data = np.load('annot_3d.npy', encoding='latin1', allow_pickle=True).item() 

print(data)
for key in data.keys():
    #print('v SHAPE-------',data[key]['fitting_params']['pose'].shape)
    #print('pose SHAPE-------',data[key]['fitting_params']['pose'].reshape(-1,3))
    print('SHAPE----------',data[key]['p2d'].shape)
    break
