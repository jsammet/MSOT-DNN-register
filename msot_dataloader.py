class Dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, mri_data, msot_data, msot_label, file_list, inshape):
        'Initialization'
        self.list_IDs = list_IDs
        self.fixed_image = mri_data[0].tensor
        print("Fixed, MRI image:")
        plt.figure()
        print("Shape: ",self.fixed_image[0,:,35,:].shape )
        plt.imshow(self.fixed_image[0,:,35,:], cmap='gray')
        self.shape_ = inshape #tripel
        self.msot_data = msot_data
        self.msot_label = msot_label
        self.file_list = file_list
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        moving_image = self.msot_data[ID].tensor 
        correct_image = self.msot_label[ID].tensor
        image_name = self.file_list[ID]
        
        return moving_image, self.fixed_image, correct_image, image_name
