
"""
Created on Sat Nov  20 23:10:18 2023

@authors: Ikram EL HAZDOUR and Michel LE PAGE
CESBIO IRD
"""
'''
LAI GEE Python algorithm (Based on the BVNNET model)
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

startDate = ee.Date('2023-11-20')
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
   #albedo = bblue.add(bred).add(bnir).add(bswir).add(bswir2).subtract(0.0018)
   albedo = bblue.multiply(0).add(0.13)
   albedo = albedo.rename('albedo')
   NDVI = opticalBands.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
   #Compute EVI for L8
   NIR =  opticalBands.select('SR_B5')
   RED =  opticalBands.select('SR_B4')
   BLUE = opticalBands.select('SR_B2')
   EVI = ((NIR.subtract(RED)).divide(NIR.add(RED.multiply(6)).subtract(BLUE.multiply(7.5)).add(1))).multiply(2.5)
   EVI = EVI.rename('EVI')

   scaled_image = (
        image.addBands(opticalBands, None, True)
             .addBands(thermalBands, None, True)
             .addBands(Tr).addBands(albedo).addBands(NDVI).addBands(EVI) #.addBands(F_g)
    )

   return scaled_image

print('3')

#=================Scale factors L5L7====================================
"""
Scale Landsat surface reflectance data
Select and rename the radiometric temperature (B6)
Compute shortwave albedo using the empirical coefficients 
from Liang et al. (2001)
Compute NDVI 
"""
def applyScaleFactorsL5L7(image):
   opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
   thermalBands = image.select('ST_B6').multiply(0.00341802).add(149.0)
   Tr = thermalBands.rename('radiometric_temperature')
   # ============Liang et al., 2001==================================
   bblue = opticalBands.select('SR_B1').multiply(0.356)
   bred = opticalBands.select('SR_B3').multiply(0.130)
   bnir = opticalBands.select('SR_B4').multiply(0.373)
   bswir = opticalBands.select('SR_B5').multiply(0.085)
   bswir2 = opticalBands.select('SR_B7').multiply(0.072)
  # albedo = bblue.add(bred).add(bnir).add(bswir).add(bswir2).subtract(0.0018)
   albedo = bblue.multiply(0).add(0.13)
   albedo = albedo.rename('albedo')
   NDVI = opticalBands.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI')
   #Compute EVI for L5
   #EVI = 2.5 * ((Band 4 – Band 3) / (Band 4 + 6 * Band 3 – 7.5 * Band 1 + 1)). 
   NIR =  opticalBands.select('SR_B4')
   RED =  opticalBands.select('SR_B3')
   BLUE = opticalBands.select('SR_B1')
   EVI = ((NIR.subtract(RED)).divide(NIR.add(RED.multiply(6)).subtract(BLUE.multiply(7.5)).add(1))).multiply(2.5)
   EVI = EVI.rename('EVI')
   scaled_image1 = (
       image.addBands(opticalBands, overwrite=True) \
                 .addBands(thermalBands, overwrite=True) \
                 .addBands(Tr).addBands(albedo).addBands(NDVI).addBands(EVI) #.addBands(F_g)
    )
 
   return scaled_image1

print('4')

# ****harmonize tm and etm+ to oli******
def tm2oli (tm) :
   slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949])
   itcp = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029])
   y = tm.select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7'],['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']).set('system:time_start', tm.get('system:time_start'))
   return y
print('6')
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
merged_TOA = L8_T1_TOA.merge(L9_T1_TOA).merge(L5_T1_TOA)#.merge(L7_T1_TOA)
print('8')
#===================LANDSAT 8 & 9 T1_L2================
#======================================================
# Filter T1_L2 collection
L5_T1_L2 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            .filterDate(startDate, endDate)
            .filterBounds(geometry)
            .map(applyScaleFactorsL5L7)
            .map(maskL7)
            .map(tm2oli))


L7_T1_L2 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
            .filterDate(startDate, endDate)
            .filterBounds(geometry)
            .map(applyScaleFactorsL5L7)
            .map(maskL7)
            .map(tm2oli))

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
merged_L2 = L8_T1_L2.merge(L9_T1_L2).merge(L5_T1_L2)#.merge(L7_T1_L2)
merged_L2 = merged_L2.map(add_datetime).map(add_datetime_day)
print('9')

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
#lai = lai.map(canopyheight)


# ---------------------------------------------Export-----------------------------------------
## -----------------------------------------------------------------------------------------------

# Iterate over each image in the collection and export
et_inputs_list = lai.toList(lai.size())

num_images = et_inputs_list.size().getInfo()
print(num_images)

for i in range(num_images):
    image = ee.Image(et_inputs_list.get(i))
    task = ee.batch.Export.image.toAsset(image=image,
                                          description=f'LAI_{i}',
                                          assetId=f'projects/ee-ikramelhazdour/assets/LAI_Collection/LAI_{i}',
                                          region=geometry,
                                          scale=30)
    task.start()


