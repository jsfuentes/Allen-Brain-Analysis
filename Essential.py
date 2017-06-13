
# coding: utf-8

# In[21]:

import os, matplotlib, json, gc, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage, misc, stats 
matplotlib.style.use('ggplot')


# In[22]:

def filterdirectory(path,extension):
    """return list of files under the directory given by the path that ends with the extension"""
    files = [file for file in os.listdir(path) if file.lower().endswith(extension) and file[0] !='.']
    return files


# In[23]:

def mkdirsafe (newpath):
    """make directory if it doesn't already exist"""
    if not os.path.exists(newpath): os.makedirs(newpath)


# In[24]:

#unessential
class detectionobject:
    
    def detectlabels (self,array):
        """returns array where identical pixels are given same label"""
        labeled = measure.label(array)
        return labeled
    
    def detectobjects (self,labels):
        """returns the minimal parallelepiped that contains all of one label in slices"""
        objects = ndimage.measurements.find_objects(labels)
        return objects

    def arealist (self,array,objects):
        """returns list of the sizes of objects in same order"""
        areas = []
        [areas.append(array[obj].size) for obj in objects]
        return areas
    
    def __init__(self, array):
        """creates labels, objects, list of areas, largest area slice, and array largest area"""
        self.labeled = self.detectlabels (array)
        self.objects = self.detectobjects(self.labeled)
        self.objectareas = self.arealist(array, self.objects)
        #== statement returns an boolean array where the only spot that is true is the max area
        #[0][0] gets location of max(since its the first/only nonzero) so you get index of object of largest area
        self.largestobjectslice = self.objects[np.nonzero(np.array(self.objectareas)== np.array(self.objectareas).max())[0][0]]
        #gets actual array values of object slice
        self.largestobjectarray = array[self.largestobjectslice]


# In[25]:

def filterImage(imagearray):
    """take a guassian filter of the image to reduce noise then 
        return a boolean mask noting extremely high values"""
    #the guassian filter is used to reduce noise by blurring/smoothing the image
    imagearray = ndimage.filters.gaussian_filter(imagearray,2)
    #thresh is a boolean array mirroring the image true if the pixel value is 4.5 std's bigger than the mean
    thresh = imagearray>(imagearray.mean() + 4.5*imagearray.std())
    #imagearray[np.invert(thresh)] = 0
    #imagearray[thresh] = imagearray[thresh]>(imagearray[thresh].mean() + 1.5*imagearray[thresh].std()) 
    #return  imagearray
    if thresh.sum()<30: #if there were only a few numbers over thresh then lower standards
        thresh = imagearray>(imagearray.mean() + 3.5*imagearray.std())
    return thresh


# In[26]:

def filterObjects(labels,lower=20,upper=500):
    """returns the labels and the objects within the threshold of size"""
    objects = ndimage.measurements.find_objects(labels)
    selectors=[]
    remove_objects =[]
    #keep_objects=[]
    #selectors = a mask for slices sized between the lower and upper bound
    #this is NOT how you use list comprehension, im annoyed
    [selectors.append(labels[obj].size>lower and labels[obj].size<upper) for obj in objects]
    selectors=np.array(selectors) #turn selectors into nparray
    indexer = np.arange(selectors.size) #get two range size of selectors
    indexer_inverse = np.arange(selectors.size)
    #gets the indexes sized appropriately  
    indexer = indexer[selectors];
    #gets the indexes not sized appropriately
    indexer_inverse = indexer_inverse[np.invert(selectors)];
    #remove_objects = all the object slices not size appropriately
    [remove_objects.append(objects[o]) for o in indexer_inverse];
    #[keep_objects.append(objects[o]) for o in indexer];
    #set all the 1s to 0s in the labeled array in the slices sized inappropriately
#     np.set_printoptions(threshold='nan')#this line messes up printing should be threshold = nan apparently
    for remv in remove_objects:
        labels[remv] =0
    #find the objects in the 1-0's,
    objects = ndimage.measurements.find_objects(labels)
    #find the objects that arent none and return it and the labels(returns none when numbered labels missing) 
    keep_objects = [x for x in objects if x] 
    return labels,keep_objects 


# In[27]:

#unessential
def AutoCrop(image, skip=1):
    #sum returns the sum of the RGB values and mean averages 
    #the > returns a boolean array giving true for each entry above the average
    Sectionbinary = image.sum(2)>image.sum(2).mean()
    #run detection with the boolean array
    detectedSection = detectionobject(Sectionbinary)
    #this if seems to be some horrible form of attempted cropping and is skipped in the real code
    if skip == 0:
        if (float(np.array(detectedSection.objectareas).max())/float (np.size(Sectionbinary)))>0.20:
            return image[detectedSection.largestobjectslice]
        else:
            return image
    else:
        return image


# In[28]:

def _zoom2Large (largearray,smallarray):
    """Zoom the small array to the size of the large array"""
    largearrayshape = np.float64(largearray.shape)
    smallarrayshape = np.float64 (smallarray.shape)
    zoomfactor = (largearrayshape [0]/smallarrayshape[0],largearrayshape[1]/smallarrayshape[1])
    zoomfactor = zoomfactor #delete this line
    zoomedsmall = ndimage.zoom(smallarray, order=0, zoom = zoomfactor)
    return zoomedsmall


# In[29]:

def allencomparisonarray (sectionnumber, imagearray, Allen_detect_annotation_path='/home/dfpena/Documents/P56_Mouse_annotation/' ):
    """Read the sectionnumber from the Allen and zoom it to the size of the imagearray"""
    #read the int32 array from the panda
    Allendetected = pd.read_pickle(Allen_detect_annotation_path +'Allen_detected_annotation.panda').values[sectionnumber][0]
    largeallen = _zoom2Large(imagearray,Allendetected)
    return largeallen


# In[30]:

def filterAllenMapby(allenmap,conversionkeyword,startkeyword,identifierIter):
    """Get the conversionKeyword for the allenmap panda rows where the startkeyword of that row = the identifiers"""
    #allenmap[[s..]].v..==i.. gives a boolean array identifying the allemap rows where the startkeyword = the identifies
    #[[s..]] double brackets probably unneccessary 
    #np.nonzero(...)[0][0] gives the index of the row of the ...
    #allenmap[conv...].v...[^] gives the allenmap's conversionkeyword value of ^ index
    filtered = [allenmap[conversionkeyword].values[np.nonzero(allenmap[[startkeyword]].values==identifier)[0][0]] for identifier in identifierIter]
    return filtered


# In[31]:

def generatecelllocationarray(structureslist,sectionnumber,Regionnames,parentid,parentname):
    """Create a pandas dataframe with a row of the section number, structurelist[:,1&2], Regionnames, & parentid"""
    #parentname seems unused
    #Side is 0 if left, 1 if right side
    cellmap = pd.DataFrame({'SectionNumber': np.array([sectionnumber] * len(structureslist)),
                            'StructureID': structureslist[:,1],
                            'Side': structureslist[:,2],
                            'StructureName': Regionnames,
                            'parent_structure_id': parentid,
                            
                            
                           })
    return cellmap


# In[32]:

def enumerateRegions(objects,ResizedAllen,Sectionnumber,allenpandamappath ='/home/dfpena/Documents/P56_Mouse_annotation/P56_Mouse_annotation/'):
    """generate a pandas DataFrame recording the info about the objects from the image overlaid on the resized allen"""
    allenmap = pd.read_pickle(allenpandamappath + 'Allen_Lookup.panda')
    #lookup is a panda dataframe with atlas_id,id,name,ontology_id and parent structure 
    #recall objects are slices
    AllenRegion=[]  
    #obj[1].start gets the lowest index of the bounding box for the x/width axis 
    #stats... takes the mode(probably id naming region) of the object slices from the processing image overlayed on allen([0][0] gives single top mode)
    #AllenRegion is composed of a 2D list which each sublist is 3 elements corresponding to the start, mode, and -1 
    [AllenRegion.append([obj[1].start,stats.mode(ResizedAllen[obj].flatten())[0][0].astype('int'),-1]) for obj in objects];
    AllenRegion= np.array(AllenRegion)
    #Allen Region = all 3 elements of the list where the mode is greater than 0 and not equal to 8
    AllenRegion= AllenRegion[AllenRegion[:,1] > 0]
    AllenRegion= AllenRegion[AllenRegion[:,1] != 8]
    #range number of rows left
    selector_range = np.arange(AllenRegion.shape[0])
    #bool of rows where the object starts in the image below the width/2 
    bool_selector =  (AllenRegion[:,0] < ResizedAllen.shape[1]/2)
    #set third element(currently -1) of AllenRegion to 0 if object starts in image below the width/2, and 1 otherwise
    AllenRegion[selector_range[bool_selector ],2]= 0
    AllenRegion[selector_range[np.invert(bool_selector) ],2] = 1
    #using the AllenRegions, the variable names are well named
    Regionnames = filterAllenMapby(allenmap,'name','id',AllenRegion[:,1])
    parentid = filterAllenMapby(allenmap,'parent_structure_id','id',AllenRegion[:,1])
    parentname = filterAllenMapby(allenmap,'name','id',parentid)
    grandparentid = filterAllenMapby(allenmap,'parent_structure_id','id',parentid)
    grandparentname = filterAllenMapby(allenmap,'name','id',grandparentid)
    cellmap = generatecelllocationarray(AllenRegion,Sectionnumber,Regionnames,parentid,parentname)
    return cellmap


# In[33]:

def process_images(ipath,iterator,allenlibpath):
    """Find objects in the image and overlays the corresponding allen image
        Then, creates and saves a panda dataframe"""
    try:    
        mkdirsafe('panda')
        mkdirsafe('3d')
        mkdirsafe('arrays')
        #get sectionnumber from number before first _ in name
        Sectionnumber = int(ipath.split('_')[0])
        #TODO: autocrop actually just equivalent to opening the image rn
        image = AutoCrop(ndimage.imread(ipath))
        #name and save image
        np.savez('arrays/'+str(iterator)+'_'+str(Sectionnumber)+'_'+ 'croppedsection',image)
        image = filterImage(image[:,:,1]) #could just use grayscale
        #name and save mask of high values
        np.savez('arrays/'+str(iterator)+'_'+str(Sectionnumber)+'_'+'filtercroppedsection',image)
        labeled = detectionobject(image).labeled #turn boolean mask into numbered mask
        labeled, objects = filterObjects(labeled)
        #name and save labels within threshold
        np.savez('arrays/'+str(iterator)+'_'+str(Sectionnumber)+'_'+ 'labels',labeled)
        #mkdebug_fig('DebugImages',(str(Sectionnumber)+'_labeledfiltered'),labeled)
        #name and save allen image corresponding to the sectionnumber zoomed to the size of the image
        Allen = allencomparisonarray(Sectionnumber,image,allenlibpath) #allen seems to be grayscale 
        np.savez('arrays/'+str(iterator)+'_'+str(Sectionnumber)+'_'+'AllenResized',Allen)
            #mkdebug_fig('DebugImages',str(iterator) + '_' +(str(Sectionnumber)+'_Verification'),Allen,0.6)
        #generatecelllocationarray cellmap
        cellmap =enumerateRegions(objects,Allen,Sectionnumber,allenlibpath)
        #save the objects and cellmap, print and return the iterator and the path to the image
        cellmap.to_pickle('panda/'+ str(iterator) + '_' +str(Sectionnumber)+'.panda')
        np.savez('3d/'+str(iterator)+'_'+str(Sectionnumber),objects )
        print(iterator, ipath)
        return (iterator, ipath)
    except ValueError:
        print('No cells could be detected')
        print(iterator, ipath)
        return (iterator, ipath)


# In[34]:

def Collect_Pandas(directorylist):
    """Create single container of all the pickled pandas in the directory"""
    container = pd.DataFrame()
    container = container.append([pd.read_pickle(file) for file in directorylist])
    return container


# In[35]:

def mkexcel( path_to_pandas, animalname): #need xls lib
    """gather all the panda dataframes and create excel sheets and pickles total and for each side"""
    os.chdir(path_to_pandas)
    directory = filterdirectory(os.curdir,".panda")

    #get all the pandas and divide them in the left and right side
    Brain = Collect_Pandas(directory)
    LateralizedDF = Brain.groupby(['Side'])
    LeftDF = LateralizedDF.get_group(0)
    RightDF = LateralizedDF.get_group(1)
    #count the unique structure names on each side and total 
    TotalCountsDF = pd.DataFrame({'Cells':Brain.StructureName.value_counts()})
    LeftCountsDF = pd.DataFrame({'Cells':LeftDF.StructureName.value_counts()})
    RightCountsDF = pd.DataFrame({'Cells':RightDF.StructureName.value_counts()})
    
    os.chdir('..')
    mkdirsafe('ExcelSheets')
    #Raw
    Brain.to_pickle('ExcelSheets/'+animalname +'_total.compile')
    Brain.to_excel('ExcelSheets/'+animalname +'_RawTotal.xls')
    LeftDF.to_excel ('ExcelSheets/'+animalname +'_RawLeft.xls')
    RightDF.to_excel ('ExcelSheets/'+animalname +'_RawRight.xls')

    #Compiled
    TotalCountsDF.to_excel ('ExcelSheets/'+animalname +'_TotalFreq.xls')
    LeftCountsDF.to_excel ('ExcelSheets/'+animalname +'_LeftFreq.xls')
    RightCountsDF.to_excel ('ExcelSheets/'+animalname +'_RightFreq.xls')


# In[36]:

def mkmpld3figs (path_to_ExcelSheetsfolder,animalname): #need mpld3 and xlrd lib
    """make a cool html bar graph of excel sheet"""
    import mpld3
    from mpld3 import plugins
    os.chdir(path_to_ExcelSheetsfolder)
    TotalCountsDF = pd.read_excel(animalname+'_TotalFreq.xls')
    fig = plt.figure(figsize=(10,5))
    bars = plt.bar(np.arange(TotalCountsDF.size),TotalCountsDF.values)
    for i, bar in enumerate(bars.get_children()):
        tooltip = mpld3.plugins.LineLabelTooltip(bar, label=str(TotalCountsDF.index[i]))
        mpld3.plugins.connect(plt.gcf(), tooltip)
    plt.xlim(0,50)
    plt.title('Cell Detection for : Subject {0}'.format(animalname))
    plt.ylabel('Cell Count')
    plt.xlabel('Region number (Scroll over Bar for Region Name)')

    a = mpld3.fig_to_html(fig)


    Html_file= open("{0}.html".format(animalname),"w")
    Html_file.write(a)
    Html_file.close()
    plt.savefig('{0}.svg'.format(animalname))


# In[37]:

def mkdebug_fig(folder,name,fig, alphav=1):
    """save fig to folder/name"""
    mkdirsafe(folder)
    plt.imshow(fig,alpha=alphav)
    plt.savefig(folder+'/'+name)


# In[38]:

def figure_from_pickle(picklepath):
    """Make a image from all the saved arrays"""
    #npz's load a dict 
    figure = np.load(picklepath)['arr_0']
    name = picklepath.split('.')
    mkdebug_fig('DebugImages',(name[0]),figure)
    plt.close()
    #cool garbage collection
    gc.collect()


# In[39]:

def easyRun(animalname, path_to_images, path_to_allenlib):
    path_to_figurearray= path_to_images + 'arrays/'

    os.chdir(path_to_images) 
    directory = filterdirectory(os.curdir,".jpg")
    print(directory)

    i=0
    for img in directory:
        process_images(img, i, path_to_allenlib)
        i+=1

    #make an excel sheet and html figures
    mkexcel(path_to_images +'panda', animalname)
    #beyond here is just figures and debugging stuff
    mkmpld3figs(path_to_images +'ExcelSheets',animalname)

    os.chdir(path_to_figurearray)
    pickledirectory = filterdirectory(os.curdir,".npz")
    #to debug look at these images
    for file in pickledirectory:
        figure_from_pickle(file)


# In[42]:

#change these arguments 
animalname = '18'
path_to_images = 'C:/Users/Student/Desktop/LAB/scans/testbed/'
path_to_allenlib = 'C:/Users/Student/Desktop/LAB/Allen/'
#for command line running in script version
if len(sys.argv) == 4:
    print("Using command line arguments")
    animalname = sys.argv[1]
    path_to_images = sys.argv[2]
    path_to_allenlib = sys.argv[3]
    
easyRun(animalname, path_to_images, path_to_allenlib)


# In[ ]:




# In[ ]:



