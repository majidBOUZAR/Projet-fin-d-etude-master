import streamlit as st
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras


def save_uploaded_file(uploadedfile):
  with open(os.path.join(uploadedfile.name),"wb") as f:
     f.write(uploadedfile.getbuffer())
  return st.success("Saved file :{} in tempDir".format(uploadedfile.name))

st.title('Segmentation des tumeurs cérébrales')
st.write('mode developpement')
datafile = st.file_uploader("Upload nii file",type=['nii.gz'])
if datafile is not None:
    file_details = {"FileName":datafile.name,"FileType":datafile.type}
    save_uploaded_file(datafile)

#######################################################################

#select uploaded file
def file_selector(folder_path='.'):
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".nii") or f.endswith(".gz")]
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    if selected_filename is None:
        return None
    return os.path.join(folder_path, selected_filename)


def plot_zslice(vol, z):
    fig, ax = plt.subplots()
    plt.axis('off')
    zslice = vol[:, :, z]
    ax.imshow(zslice.T, origin='lower', cmap='gray')
    return fig


filename = file_selector()
vis_img = st.checkbox('Show Uploaded Images')
   

if vis_img:
    img = nib.load(filename)
    img_data = img.get_fdata()
    zlen = img_data.shape[2]
    #z = st.slider('Slice', 0, zlen, int(zlen/2))
    fig = plot_zslice(img_data, 100)
    plot = st.pyplot(fig)
####################################################################
VOLUME_SLICES = 100
IMG_SIZE = 128
VOLUME_START_AT = 40

SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'Whole Tumor', # or NON-ENHANCING tumor CORE
    2 : 'Core Tumor',
    3 : 'Enhacing Tumor' # original 4 -> converted into 3 later
}

model = keras.models.load_model('./model_x1_1.h5',compile=False)

#######################################################################
plt.rcParams.update({'font.size': 22})
def predictByPath():
    #files = next(os.walk(case_path))[2]
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    #y = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE))
    
    vol_path = os.path.join(filename)
    flair=nib.load(vol_path).get_fdata()
    

    for j in range(100):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
    
    return model.predict(X, verbose=1)
    
    
def showPredictsById(start_slice = 60):
    #gt = nib.load(os.path.join(path, f'BraTS18_Training_001_seg.nii.gz')).get_fdata()
    origImage = nib.load(os.path.join(filename)).get_fdata()
    p = predictByPath()

    core = p[:,:,:,1]
    edema= p[:,:,:,2]
    enhancing = p[:,:,:,3]

    plt.figure(figsize=(50, 50))
    f, axarr = plt.subplots(1,5, figsize = (50, 50)) 

    for i in range(5): # for each image, add brain background
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
    
    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text(filename)

    axarr[1].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[1].title.set_text('Tous les classes combinés') 

    axarr[2].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[2].title.set_text(f'{SEGMENT_CLASSES[1]}')

    axarr[3].imshow(core[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[2]}')

    axarr[4].imshow(enhancing[start_slice,:,], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[3]}')
    
    #plt.show()
    plt.savefig("mygraph.png")
    st.pyplot(plt)

    

########################################################################    
    
if st.button('Predict'):
    showPredictsById()
