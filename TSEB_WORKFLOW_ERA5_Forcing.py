
"""
Created on Sat Nov  20 23:10:18 2023

@authors: Ikram EL HAZDOUR and Michel LE PAGE
"""
'''
TSEB  WORKFLOW FORCED BY ERA5 DATA
'''
"""

"""

import ee
import geemap
import numpy as np
import math


ee.Initialize()

# Add a datetime property to the Landsat data in the same format as ECMWF
def add_datetime(img):
    d = img.get('system:time_start')
    d_parsed = ee.Date(d).format("yyyyMMdd'T'HH")
    img = img.set('system:datetime', d_parsed)
    return img

# Add a datetime property to the Landsat data in the same format as ERA5 DAILY AGGR
def add_datetime_day(img):
    d = img.get('system:time_start')
    d_parsed = ee.Date(d).format("yyyyMMdd")
    img = img.set('system:datetime_day', d_parsed)
    return img

'''
PART I : PREPARE INPUT DATA FOR TSEB:
   1. PREPARE LANDSAT DATA 
   2. COMPUTE LAI USING THE NEURAL NETWORK AND LANDSAT BANDS
   3.COMPUTE CANOPY HEIGHT 
'''

# =================== Transform degree to radian ===========

degToRad = math.pi / 180
# ===================== GEOMETRY ===========================

geometry = ee.Geometry.Polygon(
[[[-8.65467412628975,31.42186897635193],
[-8.648312540793105,31.42214787548976],
[-8.648162337088271,31.429655039987907],
[-8.655264826559707,31.42844661019338]]], None, False)


# ================== DATE =============================

startDate = ee.Date('2016-01-01')
endDate = ee.Date('2023-12-31')

# ================== CLOUD MASKs ========================
    #Cloud mask for L8-9------------------------------\
def maskL8sr(image):
    """
    # Bit 0 - Fill
    # Bit 1 - Dilated Cloud
    # Bit 2 - Cirrus
    # Bit 3 - Cloud
    # Bit 4 - Cloud Shadow
    # Bit 5 - Snow
    """
    qaMask = image.select(['QA_PIXEL']).bitwise_and(int('111111', 2)).eq(0)
    saturationMask = image.select("QA_RADSAT").eq(0)

    # Replace the original bands with the scaled ones and apply the masks.
    return image.updateMask(qaMask).updateMask(saturationMask)
print('1')
    #Cloud mask for L5-7------------------------------\
def maskL7(img):
    qa = img.select('QA_PIXEL')
    mask = qa.bitwiseAnd(8).eq(0) \
                .And(qa.bitwiseAnd(4).eq(0)) \
                .And(qa.bitwiseAnd(16).eq(0)) \
                .And(qa.bitwiseAnd(32).eq(0))
    mask2 = img.select("QA_RADSAT").eq(0)              

  # This gets rid of irritating fixed-pattern noise at the edge of the images.
    mask3 = img.select('SR_B.*').gt(0).reduce(ee.Reducer.min())
  
  # this does not mask all bad pixels
  # var maskObersaturation = image.select('radsat_qa').eq(0)

    maskObersaturation = img.select(['SR_B1', 'SR_B3', 'SR_B4']) \
       .reduce(ee.Reducer.max()).lt(8000) \
       .focal_min(90, 'square', 'meters')
               
    return img.updateMask(mask).updateMask(mask2).updateMask(mask3)
print('2')

# Function to mask out bad LST data based on the ST_QA band
## NB : ST_QA = Surface temperature uncertainty (Kelvin) 
def maskBadLSTL8(image):

    qa = image.select('ST_QA').multiply(0.01)
  #  mask = qa.bitwiseAnd(1 << 0).eq(0)
    mask = qa.gte(2)
    return image.updateMask(mask).select('ST_B10')
def maskBadLSTL5(image):

    qa = image.select('ST_QA').multiply(0.01)
  #  mask = qa.bitwiseAnd(1 << 0).eq(0)
    mask = qa.gte(2)
    return image.updateMask(mask).select('ST_B6')

#===============Scale factors L8L9======================================
'''
   Scale Landsat surface reflectance data
   Select and rename the radiometric temperature (B10)
   Compute shortwave albedo using the empirical coefficients 
   from Liang et al. (2001)
   Compute NDVI 
'''
   
def applyScaleFactorsL8L9(image):
   opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
   thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

   Tr = thermalBands.select('ST_B10').rename('radiometric_temperature')
   # ============Liang et al., 2001==================================
   bblue = opticalBands.select('SR_B2').multiply(0.356)
   bred = opticalBands.select('SR_B4').multiply(0.130)
   bnir = opticalBands.select('SR_B5').multiply(0.373)
   bswir = opticalBands.select('SR_B6').multiply(0.085)
   bswir2 = opticalBands.select('SR_B7').multiply(0.072)
   albedo = bblue.add(bred).add(bnir).add(bswir).add(bswir2).subtract(0.0018)
   albedo = albedo.rename('albedo')
   NDVI = opticalBands.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
   #Compute EVI for L8
   #EVI = 2.5 * ((Band 5 – Band 4) / (Band 5 + 6 * Band 4 – 7.5 * Band 2 + 1)).
   NIR =  opticalBands.select('SR_B5')
   RED =  opticalBands.select('SR_B4')
   BLUE = opticalBands.select('SR_B2')
   EVI = ((NIR.subtract(RED)).divide(NIR.add(RED.multiply(6)).subtract(BLUE.multiply(7.5)).add(1))).multiply(2.5)
   EVI = EVI.rename('EVI')
   # F_g = (EVI.divide(NDVI)).multiply(1.2)
   # F_g = F_g.rename('Fg')
   # F_g = F_g.min(1).max(0)
   #=============da Silva et al., 2016========================
   # bblue = opticalBands.select('SR_B2').multiply(0.300)
   # bgreen = opticalBands.select('SR_B3').multiply(0.277)
   # bred = opticalBands.select('SR_B4').multiply(0.233)
   # bnir = opticalBands.select('SR_B5').multiply(0.143)
   # bswir = opticalBands.select('SR_B6').multiply(0.036)
   # bswir2 = opticalBands.select('SR_B7').multiply(0.012)
   #  #albedo = (0.300*SR_B2) + (0.277*SR_B3)+ (0.233*SR_B4)+ (0.143*SR_B5)+ (0.036*SR_B6)+ (0.012*SR_B7)
   # albedo = bblue.add(bgreen).add(bred).add(bnir).add(bswir).add(bswir2)
   # albedo = albedo.rename('albedo')
   scaled_image = (
        image.addBands(opticalBands, None, True)
             .addBands(thermalBands, None, True)
             .addBands(Tr).addBands(albedo).addBands(NDVI).addBands(EVI) #.addBands(F_g)
    )

   return scaled_image

print('3')

#=================Scale factors L5L7 + Harmonization of TM & ETM to OLI==============================
"""
Scale Landsat surface reflectance data
Select and rename the radiometric temperature (B6)
Compute shortwave albedo using the empirical coefficients 
from Liang et al. (2001)
Compute NDVI 
"""
def applyScaleFactorsL5L7(image) :
  #scale factors 
   opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
   thermalBands = image.select('ST_B6').multiply(0.00341802).add(149.0);
   Tr = thermalBands.rename('radiometric_temperature')
   # Edges mask
   mask3 = opticalBands.gt(0).reduce(ee.Reducer.min())
   opticalBands = opticalBands.updateMask(mask3)
  #rename bands so that L7 & L5 match L8
   renamedBands = opticalBands.select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7'], 
                                      ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
  
  #*****Harmonization****************************************************************
#(RMA method)_Roy et al., 2016
   slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949])
   itcps = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029])
# OLS method coeffs.
   slopes = ee.Image.constant([0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071])
   itcps = ee.Image.constant ([0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172])
  
   harmonizedBands = renamedBands.multiply(slopes).add(itcps)
   scaled_image1 = (
       image.addBands(thermalBands, overwrite=True) \
                 .addBands(Tr)
                 .addBands(harmonizedBands)
    )

   return scaled_image1

#=======Compute Indices outside the Scale factors function (to be mapped after the cloud mask) ;)=====
def ComputeIndicesL7 (harmonizedBands):
    # compute ALBEDO
     bblue = harmonizedBands.select('SR_B2').multiply(0.356)
     bred = harmonizedBands.select('SR_B4').multiply(0.130)
     bnir = harmonizedBands.select('SR_B5').multiply(0.373)
     bswir = harmonizedBands.select('SR_B6').multiply(0.085)
     bswir2 = harmonizedBands.select('SR_B7').multiply(0.072)
    
     albedo = bblue.add(bred).add(bnir).add(bswir).add(bswir2).subtract(0.0018)
     albedo = albedo.rename('albedo')
     #NDVI
     NDVI = harmonizedBands.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
     #FIPAR
     FIPAR = NDVI.multiply(1).add(-0.05).rename('fipar')
     NIR = harmonizedBands.select('SR_B5')
     RED = harmonizedBands.select('SR_B4')
     BLUE = harmonizedBands.select('SR_B2')
     #EVI
     EVI = ((NIR.subtract(RED)).divide(NIR.add(RED.multiply(6))
                               .subtract(BLUE.multiply(7.5))
                               .add(1))).multiply(2.5).rename('EVI')
     
     scaled_image2 = (
         harmonizedBands.addBands(albedo).addBands(NDVI).addBands(EVI).addBands(FIPAR)
      )
     return scaled_image2


# ================== RENAME BANDS ======================
def rename_bands_landsat(image):
    return image.select(['VAA', 'VZA', 'SAA', 'SZA'], ['viewAzimuthMean', 'viewZenithMean', 'sunAzimuthAngles', 'sunZenithAngles']).multiply(0.01)
print('7')
# ================== LANDSAT COLLECTIONS [TOA] =================
L5_T1_TOA = (ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
             .filterDate(startDate, endDate)
             .filterBounds(geometry)
             .map(maskL7)
             .map(rename_bands_landsat))
L5_T1_TOA = L5_T1_TOA.select(['viewAzimuthMean', 'viewZenithMean', 'sunAzimuthAngles', 'sunZenithAngles'])

#L5_T1_TOA = L5_T1_TOA.map(lambda img: img.multiply(0.01))

L7_T1_TOA = (ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA")
             .filterDate(startDate, endDate)
             .filterBounds(geometry)
             .map(maskL7)
             .map(rename_bands_landsat))
L7_T1_TOA = L7_T1_TOA.select(['viewAzimuthMean', 'viewZenithMean', 'sunAzimuthAngles', 'sunZenithAngles'])
#L7_T1_TOA = L7_T1_TOA.map(lambda img: img.multiply(0.01))

L8_T1_TOA = (ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
             .filterDate(startDate, endDate)
             .filterBounds(geometry)
             .map(maskL8sr)
             .map(rename_bands_landsat))
L8_T1_TOA = L8_T1_TOA.select(['viewAzimuthMean', 'viewZenithMean', 'sunAzimuthAngles', 'sunZenithAngles'])
#L8_T1_TOA = L8_T1_TOA.map(lambda img: img.multiply(0.01))

L9_T1_TOA = (ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA")
             .filterDate(startDate, endDate)
             .filterBounds(geometry)
             .map(maskL8sr)
             .map(rename_bands_landsat))
L9_T1_TOA = L9_T1_TOA.select(['viewAzimuthMean', 'viewZenithMean', 'sunAzimuthAngles', 'sunZenithAngles'])
#L9_T1_TOA = L9_T1_TOA.map(lambda img: img.multiply(0.01))

#=====MERGE TOA COLLECTIONS ============================
merged_TOA = L8_T1_TOA.merge(L9_T1_TOA).merge(L5_T1_TOA).merge(L7_T1_TOA)
print('8')
#===================LANDSAT 8 & 9 T1_L2================
#======================================================
# Filter T1_L2 collection
L5_T1_L2 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            .filterDate(startDate, endDate)
            .filterBounds(geometry)
            .map(applyScaleFactorsL5L7)
            .map(maskL7)
            .map(ComputeIndicesL7))


L7_T1_L2 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
            .filterDate(startDate, endDate)
            .filterBounds(geometry)
            .map(applyScaleFactorsL5L7)
            .map(maskL7)
            .map(ComputeIndicesL7))

L8_T1_L2 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .filterDate(startDate, endDate)
              .filterBounds(geometry)
              .map(applyScaleFactorsL8L9)
              .map(maskL8sr))


  
L9_T1_L2 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
              .filterDate(startDate, endDate)
              .filterBounds(geometry)
              .map(applyScaleFactorsL8L9)
              .map(maskL8sr))
              



#==========Merge L8 and L9 T1_L2 collections===========
merged_L2 = L8_T1_L2.merge(L9_T1_L2).merge(L5_T1_L2).merge(L7_T1_L2)
merged_L2 = merged_L2.map(add_datetime).map(add_datetime_day)
print('9')

#================================================================
#                        COMPUTE CANOPY HEIGHT
#================================================================
def canopyheight(image):

   classification = ee.ImageCollection('projects/ee-ikramelhazdour/assets/CLASSIF_MAPS')


   '''
    Objectif : to take each class ID (label), 
    then retrieve the corresponding CH (min & max) value from the dict above 
    Then , convert it to an ee.Number.
    The result = a new list or object (class_CHmin & class_CHmax) containing the CHmin & max values for each pixel.
    
    '''
   '''
    0 = Vineyard 
    1 = Bare Ground
    2 = Green peas
    3 = Olive trees
    4 = Double-Crops
    5 = Alfalfa
    6 = Faba beans
    7 = Cereals
    8 = Citrus trees
    9 = Apricot trees
    10 = Watermelon 
    11 = Water
    12 = Rainfed crops 
   '''

   NDVI = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
   datebis =  ee.Date(image.get('system:time_start')).advance(4, 'month')
   year = ee.Date(image.get('system:time_start')).get('year').format()
   classification = ee.ImageCollection('projects/ee-ikramelhazdour/assets/CLASSIF_MAPS')\
       .filter(ee.Filter.date(year.cat('-01-01'), year.cat('-12-31'))).first()
   
  # ====== resampling Classif maps from 10m to 30m=====================
   classification30 = classification.reduceResolution(
      reducer = ee.Reducer.mode(),
      maxPixels = 1024
    ).reproject(
      crs = NDVI.projection().atScale(30)
    );  
   #classification = ee.ImageCollection('projects/ee-ikramelhazdour/assets/CLASSIF_MAPS').first()
   
   CHmin =   classification30.remap([0,1,2,3,4,5,6,7,8,9,10,11,12], [1.5, 0.0,0.0,3.0,0.0,0.0 ,0.0,0.0,3.0,3.0,0.0,0.0,0.0])
   CHmax =   classification30.remap([0,1,2,3,4,5,6,7,8,9,10,11,12], [1.5, 0.1,0.3,3.0,0.6,0.6,0.6,0.8,3.0,3.0,0.3,0.0,0.6])
  # NDVImin = classification30.remap([0,1,2,3,4,5,6,7,8,9,10,11,12], [0.08,0.05,0.06 ,0.08,0.07,0.07,0.07,0.07,0.10,0.10,0.02,0.0,0.07])
 #NDVI MIN = 0.14 all the crops
   NDVImin = classification30.remap([0,1,2,3,4,5,6,7,8,9,10,11,12], [0.14,0.14,0.14 ,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.0,0.14])
   NDVImax = classification30.remap([0,1,2,3,4,5,6,7,8,9,10,11,12], [0.63,0.36,0.63,0.73,0.71,0.78,0.79,0.78,0.74,0.73,0.69,0.0,0.62])
    
    # Compute scaled canopy height (CH) for each class
   CH = CHmin.add((CHmax.subtract(CHmin)).multiply(NDVI.subtract(NDVImin)).divide(NDVImax.subtract(NDVImin))).rename('canopy_height')
   CH = CH.max(0).min(3.5) # on seuille entre 0 et 3.5 m
    # Add computed bands to the image
   scaled_image = image.addBands(CH)

   return scaled_image


def setdate(image):
    img1 = ee.Image(image.get('primary'))
    return(image.set('system:time_start', img1.get('system:time_start')))


# Filter collections by date
filterDate = ee.Filter.equals(
  leftField = 'system:index',
  rightField = 'system:index'
)

#join
Join = ee.Join.inner()

joined_data = ee.ImageCollection(Join.apply(merged_L2, merged_TOA, filterDate))
#joined_data = joined_data.map(setdate)
print('11')
#=======Flatten the joined coll============

def reduceNesting(f) :  
    return ee.Image(f.get('primary')).addBands(f.get('secondary'))
    #return(f.get('primary')).addBands(f.get('secondary'))

joined = joined_data#.map(reduceNesting)
#print(joined.first().get('system:time_start'))

print('12')
#==========================================================
#                   COMPUTE LEAF AREA INDEX
#==========================================================

# ==================1.NORMALIZATION FUNCTIONS ==============
# Function to normalize bands
def normalize(image, band, min_val, max_val):
    return ((image.select([band]).subtract(min_val)).multiply(2)).divide(max_val - min_val).subtract(1)
print('13')

# Function to normalize cosine bands
def normalize_cos(image, band, min_val, max_val):
    coss = image.select([band]).multiply(degToRad).cos()
    return coss.subtract(min_val).multiply(2).divide(max_val - min_val).subtract(1)
print('14')

# Function to denormalize bands
def denormalize(image, band, min_val, max_val):
    return ((image.select([band]).add(1)).multiply(0.5)).multiply(max_val - min_val).add(min_val)
print('15')

#Function to calculate tanh sigmoid
# 2 / (1 + Math.exp(-2 * input)) - 1; 
def tansig(img) :
   a = (img.multiply(-2)).exp().add(1)
   b = a.multiply(0).add(2).divide(a).subtract(1)
  #Map.addLayer(b.select('B3'), {min: 0, max: 5}, 'tansig');
   return b
  #return img.multiply(-2).exp().add(1).divide(img.multiply(-2).exp().subtract(1)).multiply(2);
print('16')

# ================== NEURAL NETWORK FUNCTIONS ============

def computeLAI(image) :
    
  primary_image = ee.Image(image.get('primary'))
  secondary_image = ee.Image(image.get('secondary'))
  ndvi = primary_image.select('NDVI')
 
  b03_norm = normalize(primary_image, 'SR_B3', 0, 0.253368175)
  b04_norm = normalize(primary_image, 'SR_B4', 0, 0.280707189)
 # var b05_norm = normalize(image, 'SR_B5', 0, 0.305398915248555);
 # var b06_norm = normalize(image, 'B6', 0.006637972542253, 0.608900395797889);
 # var b07_norm = normalize(image, 'B7', 0.013972727018939, 0.753827384322927);
  b8a_norm = normalize(primary_image, 'SR_B5', 0.024524742, 0.792621915)
  b11_norm = normalize(primary_image, 'SR_B6', 0.001975198, 0.490593793)
  b12_norm = normalize(primary_image, 'SR_B7', 0, 0.488251517)

   # Map.addLayer(image.select('SR_B3'), {min: 0, max: 5}, 'b03_');
   # Map.addLayer(b03_norm.select('SR_B3'), {min: 0, max: 5}, 'b03_norm');
  
  viewZen_norm = normalize_cos(secondary_image, 'viewZenithMean', 0.989440127, 1)
  sunZen_norm = normalize_cos(secondary_image, 'sunZenithAngles', 0.342281984, 0.922571154)
  relAzim_norm = (secondary_image.select('sunAzimuthAngles').subtract(secondary_image.select('viewAzimuthMean'))).multiply(degToRad).cos()


   # Map.addLayer(image.select('viewZenithMean'), {min: 0, max: 5}, 'viewZenithMean')
   # Map.addLayer(viewZen_norm, {min: 0, max: 5}, 'viewZenithNorm')

  def neuron1(b03_norm,b04_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) :
    sum1 = b03_norm.multiply(1.440327565).add(b04_norm.multiply(-0.482190959)).add(b8a_norm.multiply(1.218821198)).add(b11_norm.multiply(1.371645993)).add(b12_norm.multiply(0.710114786)).add(viewZen_norm.multiply(0.064438242)).add(sunZen_norm.multiply(-0.350301091)).add(relAzim_norm.multiply(-0.056380508)).add(-0.329156829)   
    return sum1.tanh()
    #return np.tanh(sum)
    
  def neuron2(b03_norm,b04_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) :
    sum1 = b03_norm.multiply(0.261715374).add(b04_norm.multiply(0.166531339)).add(b8a_norm.multiply(-1.204848164)).add(b11_norm.multiply(0.664689338)).add(b12_norm.multiply(-0.142765223)).add(viewZen_norm.multiply(0.036892839)).add(sunZen_norm.multiply(0.065248421)).add(relAzim_norm.multiply(0.015555027)).add(1.265179784)
    return sum1.tanh()


  def neuron3(b03_norm,b04_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) :
   sum1 = b03_norm.multiply(-0.650705702).add(b04_norm.multiply(-0.099681393)).add(b8a_norm.multiply(0.95511522)).add(b11_norm.multiply(-0.386112214)).add(b12_norm.multiply(-0.328887138)).add(viewZen_norm.multiply(-0.208367086)).add(sunZen_norm.multiply(0.089891761)).add(relAzim_norm.multiply(0.11672358)).add(-0.638800198)
   return sum1.tanh()


  def neuron4(b03_norm,b04_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) :
   sum1 = b03_norm.multiply(0.634535851).add(b04_norm.multiply(-0.104234441)).add(b8a_norm.multiply(-0.284511302)).add(b11_norm.multiply(-0.748279698)).add(b12_norm.multiply(0.00112764)).add(viewZen_norm.multiply(0.694009199)).add(sunZen_norm.multiply(0.575512234)).add(relAzim_norm.multiply(-0.269886963)).add(0.672954129)
   return sum1.tanh()


  def neuron5(b03_norm,b04_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) :
   sum1 = b03_norm.multiply(0.822959518).add(b04_norm.multiply(-0.102043433)).add(b8a_norm.multiply(-0.300375226)).add(b11_norm.multiply(0.630195255)).add(b12_norm.multiply(0.955350112)).add(viewZen_norm.multiply(-0.673728269)).add(sunZen_norm.multiply(-0.559327322)).add(relAzim_norm.multiply(0.184242329)).add(2.608881911)
   return sum1.tanh()
   

  def layer2(neuron1, neuron2, neuron3, neuron4, neuron5):
   sum1 = neuron1.multiply(0.107827277) \
   .add(neuron2.multiply(-1.09777748)) \
   .add(neuron3.multiply(0.186176019)) \
   .add(neuron4.multiply(0.086218419)) \
   .add(neuron5.multiply(-0.097040014)) \
   .add(0.311068524)
   return sum1



  neuron1LL = neuron1(b03_norm,b04_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
  neuron2LL = neuron2(b03_norm,b04_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
  neuron3LL = neuron3(b03_norm,b04_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
  neuron4LL = neuron4(b03_norm,b04_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
  neuron5LL = neuron5(b03_norm,b04_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)

  #Combine the results using the layer2 function
  layer2LL = layer2(neuron1LL, neuron2LL, neuron3LL, neuron4LL, neuron5LL)
 #Denormalize LAI
  lai = denormalize(layer2LL, 'SR_B3', 0.000233774, 13.83459255)
  lai = lai.max(0).min(10) # on seuille entre 0 et 10 
  
  #Add a condition based on NDVI
  lai = lai.where(ndvi.lte(0.2), 0)
  
  return primary_image.addBands(lai.rename('LAI_neural'))

lai = joined.map(computeLAI)
lai = lai.map(canopyheight)

'''
PART II : PREPARE THE METEOROLOGICAL INPUTS FOR TSEB
USING ERA5 DATA 

'''

#==================== Meteorological data ============================

## ECMWF collection 
# https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY#bands)

bands_to_keep = ee.List([\
    'surface_pressure', \
    'temperature_2m', \
    'u_component_of_wind_10m',\
    'v_component_of_wind_10m',\
    'surface_solar_radiation_downwards_hourly',\
    'surface_thermal_radiation_downwards_hourly',\
    'dewpoint_temperature_2m'
    ])
bands_to_rename = ee.List([\
    'surface_pressure',\
    'air_temperature', \
    'u_component_of_wind_10m',\
    'v_component_of_wind_10m',\
    'surface_solar_radiation_downwards_hourly',\
    'surface_thermal_radiation_downwards_hourly',\
    'dewpoint_temperature_2m'
    ])

bands_to_keep1 = ee.List([\
    'surface_solar_radiation_downwards_sum'
    ])
bands_to_rename1 = ee.List([\
    'surface_solar_radiation_downwards_sum'
    ])

def prepare_ECMWF_for_TSEB(img):
    # wind speed: square root of sum of squares
    u = img.select('u_component_of_wind_10m')
    v = img.select('v_component_of_wind_10m')
    wind = ((u.pow(2)).add(v.pow(2))).pow(0.5).rename('wind_speed')

    # solar and thermal radiation conversion to W/m2 and renaming bands
    Sdn = img.select('surface_solar_radiation_downwards_hourly').divide(3600.0).rename('solar_radiation')
    Ldn = img.select('surface_thermal_radiation_downwards_hourly').divide(3600.0).rename('thermal_radiation')
    Sdn_j = img.select('surface_solar_radiation_downwards_hourly').rename('solar_radiation_j')
    return img.addBands(wind).addBands(Sdn).addBands(Ldn).addBands(Sdn_j)

Meteo_collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
    .filterDate(startDate, endDate) \
            .select(bands_to_keep, bands_to_rename)

Meteo_collection = Meteo_collection.map(prepare_ECMWF_for_TSEB)
print('23')

# daily EF prepare data
def prepare_daily_for_TSEB (img):
    Sdn_day = img.select('surface_solar_radiation_downwards_sum')#.rename('Sdn_day')
    return img.addBands(Sdn_day)
Meteo_collection_day = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
    .filterDate(startDate, endDate) \
     .select(bands_to_keep1, bands_to_rename1)
#Meteo_collection_day = Meteo_collection_day.map(prepare_daily_for_TSEB)
#print(Meteo_collection_day)
#========================================================================================
'''
PART III: ADD A DATETIME PROPERTY TO THE LANDSAT DATA IN THE SAME FORMAT AS ECMWF
'''
joined = lai
print('24')
# Join the two collections using the datetime property. 
filterByDateTime = ee.Filter.equals(leftField='system:datetime', rightField='system:index')
joinByDateTime = ee.ImageCollection(ee.Join.inner().apply(joined, Meteo_collection, filterByDateTime))
print('25')
def get_img(feature):
    return ee.Image.cat(feature.get('primary'), feature.get('secondary'))

et_inputs = joinByDateTime.map(get_img)
##===============================================================
# Join daily ERA5 and et_inputs  to compute EF
filterByDateDay = ee.Filter.equals(leftField='system:datetime_day', rightField='system:index')
# This was incorrect
joinByDateTimeDay = et_inputs

# Correct join operation between et_inputs and Meteo_collection_day
joinByDateTimeDay = ee.ImageCollection(ee.Join.inner().apply(et_inputs, Meteo_collection_day, filterByDateDay))

def merge_features(feature):
    primary = ee.Image(feature.get('primary'))
    secondary = ee.Image(feature.get('secondary'))
    return primary.addBands(secondary)

# Map the function to merge features from the join result
et_inputs = joinByDateTimeDay.map(merge_features)

print('26')

'''
PART IV: PROVIDE PARAMETERS FOR TSEB INPUTS 
'''
# Provide local time offset (UTC+1 for MOROCCO). Required for the diurnal calculations related to 
# partitioning of solar radiation, as well as for G (diurnal variations of ground heat flux).
time_offset = 1
viewing_zenith_angle = 0 # 0 for Landsat (nadir)
zU = 10 # ECMWF: wind measured at 10m
zT = 2 # ECMWF: temperature measured at 2m
def time_properties(img):
    d = ee.Date(img.date())
    doy = d.getRelative('day','year').add(1)  # from 0-based to 1-based. 
    time = d.getRelative('hour', 'day').add(time_offset)  
    img = img.set('doy', doy)
    img = img.set('time', time)
    img = img.set('viewing_zenith', viewing_zenith_angle)
    img = img.set('zU', zU)
    img = img.set('zT', zT)
    return img
print('27')
et_inputs = et_inputs.map(time_properties)
print('28')
#print(et_inputs.getInfo(),'et coll')

print('29')
'''
PART V : EXPORT THE TSEB INPUTS AS ASSETS ON GEE
TO BE IMPORTED AND USED IN THE LAST PART OF THIS CODE
'''
# ---------------------------------------------Export-----------------------------------------
## -----------------------------------------------------------------------------------------------

# Iterate over each image in the collection and export
et_inputs_list = et_inputs.toList(et_inputs.size())

num_images = et_inputs_list.size().getInfo()
print(num_images)

for i in range(num_images):
    image = ee.Image(et_inputs_list.get(i))
    task = ee.batch.Export.image.toAsset(image=image,
                                          description=f'TSEB_inputs20212023_{i}',
                                          assetId=f'projects/ee-ikramelhazdour/assets/TSEB_inputs_04032024_6/TSEB_inputs20212023_{i}',
                                          region=geometry,
                                          scale=30)
    task.start()

#------------HERE WE START THE EXPORT OF THE OUTPUTs------------------------------

# # Load the entire collection of input images
et_inputs = ee.ImageCollection([ee.Image(f'projects/ee-ikramelhazdour/assets/TSEB_inputs_04032024_6/TSEB_inputs20212023_{i}') for i in range(num_images)])


from geeet.tseb import tseb_series
# Iterate over each image in the collection and run the TSEB model
et_outputs = et_inputs.map(lambda img: tseb_series(img, zU=10, zT=2))

# Define a function to add ETd band

from geeet.solar import rad_ratio
def add_ET_mm_band(img):
    R = rad_ratio(img) #====Jackson irradiance model====
    LE = img.select('LE') # in W/m2
    #=======Upscaling instantaneous ET to 24h ET using the Evaporative Fraction (Ryu et al., 2012))====================
    #=================================================================================================================
    Sdd= img.select('surface_solar_radiation_downwards_sum') # solar radiation daily sum in J/m2
    Sd_w = img.select('solar_radiation') # hourly solar radiation in w/m2
    G = img.select('G') # soil heat flux, neglected
    EF = LE.divide(Sd_w.subtract(G)) # a %
    LEd = EF.multiply(Sdd) # in J/m2
  #  ETd = LE.multiply(R).rename('ETd')
    ETd = LEd.divide(2.45e6).rename('ETd') # conversion from j/m2 to mm/day
    ETd = ETd.max(0).min(10)
    return img.addBands(ETd)

# Map the function over the outputs collection
et_outputs = et_outputs.map(add_ET_mm_band)

# Iterate over each output image in the collection and export
et_outputs_list = et_outputs.toList(et_outputs.size())
num_outputs = et_outputs_list.size().getInfo()

for i in range(num_outputs):
    output_image = ee.Image(et_outputs_list.get(i))
    task = ee.batch.Export.image.toAsset(image=output_image,
                                          description=f'TSEB_outputs20212023_{i}',
                                          assetId=f'projects/ee-ikramelhazdour/assets/TSEB_outputs_04032024_6/TSEB_outputs20212023_{i}',
                                          region=geometry,
                                          scale=30)
    task.start()
