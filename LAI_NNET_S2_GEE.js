//================================================================================
//Script authors : Ikram EL HAZDOUR (PhD candidate at CESBIO & UCAM),
//                 Michel LE PAGE (Researcher at CESBIO IRD)

//Version:  2023 Version
//================================================================================
//********************************************************************************
//The NNET Coef. used in this script were retreived from the following package: 
//              https://github.com/senbox-org/s2tbx/tree/master
//********************************************************************************

//****ikram : please define a geometry first

// Transform degree to radian
var degToRad = Math.PI / 180;
var S2scale = 0.0001;

//====================================================
// Functions for the neural network and normalization :)
//====================================================

// New normalize function: x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
function normalize(img, band, min, max) {
  return ((img.select(band).multiply(S2scale).subtract(min)).multiply(2)).divide(max - min).subtract(1);
}

function normalize_cos(img, band, min, max) {
  var coss = img.select(band).multiply(degToRad).cos();
  return coss.subtract(min).multiply(2).divide(max - min).subtract(1);
}

// New denormalize function: y = 0.5 * (y_norm + 1) * (y_max - y_min)
function denormalize(img, band, min, max) {
  return ((img.select(band).add(1)).multiply(0.5)).multiply(max - min);
}


// Function to calculate tanh sigmoid
// 2 / (1 + Math.exp(-2 * input)) - 1; 
function tansig(img) {
  var a = (img.multiply(-2)).exp().add(1);
  var b = a.multiply(0).add(2).divide(a).subtract(1);
  //Map.addLayer(b.select('B3'), {min: 0, max: 5}, 'tansig');

  return(b);
  //return img.multiply(-2).exp().add(1).divide(img.multiply(-2).exp().subtract(1)).multiply(2);
}

// Function to calculate neuron
/*
function neuron(image, coeffs) {
    var bands = image.bandNames();
    var weightedSum = ee.ImageCollection(bands.map(function(band, index) {
     return image.select([band]).multiply(coeffs[index + 1]).sum().add(coeffs[0])}
  return tansing(weightedSum)
}
*/
// Function to add mean angles
function addMeanAngles(image) {
  var bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9',  'B11', 'B12'];

  // Function to calculate mean of angles for a specific band
  function calculateMeanAngle(band) {
    var incidenceAngleBand = 'MEAN_INCIDENCE_AZIMUTH_ANGLE_' + band ;
    var IncidenceAngle = image.get(incidenceAngleBand);
    return ee.Number(IncidenceAngle);
  }

  function calculateMeanAngle2(band) {
    var incidenceAngleBand = 'MEAN_INCIDENCE_ZENITH_ANGLE_' + band ;
    var IncidenceAngle = image.get(incidenceAngleBand);
    return ee.Number(IncidenceAngle);
  }
  
  var solarAzimuthBand = 'MEAN_SOLAR_AZIMUTH_ANGLE';
  var mean_solar_azimuth = image.get(solarAzimuthBand);
  var solarAzimuthBand = 'MEAN_SOLAR_ZENITH_ANGLE';
  var mean_solar_zenith = image.get(solarAzimuthBand);
  
  var IncidenceAngles = bands.map(calculateMeanAngle);
  var mean_incidence_azimuth_angle=ee.List(IncidenceAngles).reduce(ee.Reducer.mean());
  
  var IncidenceAngles = bands.map(calculateMeanAngle2);
  var mean_incidence_zenith_angle=ee.List(IncidenceAngles).reduce(ee.Reducer.mean());

  //var meanSolarAzimuth = image.get(solarAzimuthBand).rename('mean_solar_azimuth');
  return(
    image.addBands([
      ee.Image.constant(mean_incidence_azimuth_angle).rename("viewAzimuthMean"),
      ee.Image.constant(mean_incidence_zenith_angle).rename("viewZenithMean"),
      ee.Image.constant(mean_solar_azimuth).rename("sunAzimuthAngles"),
      ee.Image.constant(mean_solar_zenith).rename("sunZenithAngles")]
  ));
}

// Compute LAI function 
function computeLAI(image) {
 
  var b03_norm = normalize(image, 'B3', 0, 0.253061520471542);
  var b04_norm = normalize(image, 'B4', 0, 0.290393577911328);
  var b05_norm = normalize(image, 'B5', 0, 0.305398915248555);
  var b06_norm = normalize(image, 'B6', 0.006637972542253, 0.608900395797889);
  var b07_norm = normalize(image, 'B7', 0.013972727018939, 0.753827384322927);
  var b8a_norm = normalize(image, 'B8A', 0.026690138082061, 0.782011770669178);
  var b11_norm = normalize(image, 'B11', 0.016388074192258, 0.493761397883092);
  var b12_norm = normalize(image, 'B12', 0, 0.493025984460231);
  print('b12',b12_norm)
    Map.addLayer(image.select('B3'), {min: 0, max: 5}, 'b03_');
    Map.addLayer(b03_norm.select('B3'), {min: 0, max: 5}, 'b03_norm');
  
  var viewZen_norm = normalize_cos(image, 'viewZenithMean', 0.918595400582046, 1);
  var sunZen_norm = normalize_cos(image, 'sunZenithAngles', 0.342022871159208, 0.936206429175402);
  var relAzim_norm = (image.select('sunAzimuthAngles').subtract(image.select('viewAzimuthMean'))).multiply(degToRad).cos();
  print(relAzim_norm)
    Map.addLayer(image.select('viewZenithMean'), {min: 0, max: 5}, 'viewZenithMean');
    Map.addLayer(viewZen_norm, {min: 0, max: 5}, 'viewZenithNorm');


  function neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
    var sum = b03_norm.multiply(-0.023406878966470)
    .add(b04_norm.multiply(0.921655164636366))
    .add(b05_norm.multiply(0.135576544080099))
    .add(b06_norm.multiply(-1.938331472397950))
    .add(b07_norm.multiply(-3.342495816122680))
    .add(b8a_norm.multiply(0.902277648009576))
    .add(b11_norm.multiply(0.205363538258614))
    .add(b12_norm.multiply(-0.040607844721716))
    .add(viewZen_norm.multiply(-0.083196409727092))
    .add(sunZen_norm.multiply(0.260029270773809))
    .add(relAzim_norm.multiply(0.284761567218845))
    .add(4.96238030555279);
    
    //print(sum,'sum');
    //Map.addLayer(sum.select('B3'), {min: 0, max: 5}, 'sum');
    //var a = sum.multiply(-2).exp().add(1);
    //var b = a.multiply(0).add(2).divide(a).subtract(1);
    //Map.addLayer(b.select('B3'), {min: 0, max: 5}, 'b');
    
    //var tansig1 = tansig(sum);
    //Map.addLayer(tansig1.select('B3'), {min: 0, max: 5}, 'tansig');
    //var tansig2 = sum.tanh();
    //Map.addLayer(tansig2.select('B3'), {min: 0, max: 5}, 'tanh');
    //return tansing(sum);
    return sum.tanh();
  }



function neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
   var sum = b03_norm.multiply(-0.132555480856684)
	.add(b04_norm.multiply(-0.139574837333540))
	.add(b05_norm.multiply(-1.014606016898920))
	.add(b06_norm.multiply(-1.330890038649270))
  .add(b07_norm.multiply(0.031730624503341))
  .add(b8a_norm.multiply(-1.433583541317050 ))
  .add(b11_norm.multiply(-0.959637898574699))
	.add(b12_norm.multiply(1.133115706551000))
  .add(viewZen_norm.multiply(0.216603876541632))
  .add(sunZen_norm.multiply(0.410652303762839))
  .add(relAzim_norm.multiply(0.064760155543506))
  .add(1.416008443981500);

  //return tansing(sum);
    return sum.tanh();
}

function neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
  var sum = b03_norm.multiply(0.086015977724868 )
	.add(b04_norm.multiply(0.616648776881434))
	.add(b05_norm.multiply(0.678003876446556))
	.add(b06_norm.multiply(0.141102398644968 ))
  .add(b07_norm.multiply(-0.096682206883546))
  .add(b8a_norm.multiply(-1.128832638862200))
  .add(b11_norm.multiply(0.302189102741375))
	.add(b12_norm.multiply(0.434494937299725))
  .add(viewZen_norm.multiply(-0.021903699490589))
  .add(sunZen_norm.multiply(-0.228492476802263))
  .add(relAzim_norm.multiply(-0.039460537589826 ))
  .add(1.075897047213310);

    return sum.tanh();
}


function neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
  var sum = b03_norm.multiply(-0.109366593670404)
	.add(b04_norm.multiply(-0.071046262972729))
	.add(b05_norm.multiply(0.064582411478320))
	.add(b06_norm.multiply(2.906325236823160))
  .add(b07_norm.multiply(-0.673873108979163))
  .add(b8a_norm.multiply(-3.838051868280840))
  .add(b11_norm.multiply(1.695979344531530 ))
	.add(b12_norm.multiply(0.046950296081713 ))
  .add(viewZen_norm.multiply(-0.049709652688365))
  .add(sunZen_norm.multiply(0.021829545430994))
  .add(relAzim_norm.multiply(0.057483827104091))
  .add(1.533988264655420);

    return sum.tanh();
}


function neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
  var sum = b03_norm.multiply(-0.089939416159969)
	.add(b04_norm.multiply(0.175395483106147))
	.add(b05_norm.multiply(-0.081847329172620))
	.add(b06_norm.multiply(2.219895367487790))
  .add(b07_norm.multiply(1.713873975136850))
  .add(b8a_norm.multiply(0.713069186099534))
  .add(b11_norm.multiply( 0.138970813499201))
	.add(b12_norm.multiply(-0.060771761518025))
  .add(viewZen_norm.multiply(0.124263341255473))
  .add(sunZen_norm.multiply(0.210086140404351))
  .add(relAzim_norm.multiply(-0.183878138700341))
  .add(3.024115930757230);

    return sum.tanh();
}


function layer2(neuron1, neuron2, neuron3, neuron4, neuron5) {
  var sum = neuron1.multiply(-1.500135489728730)
  .add(neuron2.multiply(-0.096283269121503))
  .add(neuron3.multiply(-0.194935930577094))
  .add(neuron4.multiply(-0.352305895755591))
	.add(neuron5.multiply(0.075107415847473))
  .add(1.096963107077220);
  return sum;
}

// 
  var neuron1LL = neuron1(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm);
  print('neuron1',neuron1LL)
  Map.addLayer(neuron1LL.select('B3'), {min: 0, max: 5}, 'neuron');

  var neuron2LL = neuron2(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm);
  var neuron3LL = neuron3(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm);
  var neuron4LL = neuron4(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm);
  var neuron5LL = neuron5(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm);
  print('neuron5',neuron5LL)

  // Combine the results using the layer2 function
   var layer2LL = layer2(neuron1LL, neuron2LL, neuron3LL, neuron4LL, neuron5LL);
  print('layer2',layer2LL)

  // Denormalize LAI
  var lai = denormalize(layer2LL, 'B3', 0.000319182538301, 14.4675094548151);
  // ****MLP: il n'y a pas de division par 3 dans https://github.com/ollinevalainen/satellitetools/blob/develop/biophys/biophys_xarray.py ni dans 
  //  http://step.esa.int/docs/extra/ATBD_S2ToolBox_L2B_V1.1.pdf
  //lai = lai.divide(3);
 // print('layer2Result', layer2Result);
 print('lai', lai);
return image.addBands(lai.rename('LAI'));
}


// Load the Sentinel-2 ImageCollection 
var sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterDate('2020-04-06', '2020-12-31').filterBounds(geometry);
//print(sentinel2.first());

sentinel2 = sentinel2.map(addMeanAngles);
print(sentinel2);
//var aaa= addMeanAngles(sentinel2.first());
//print(aaa);

//sentinel2 = sentinel2.map(computeLAI);
sentinel2=computeLAI(sentinel2.first());
print(sentinel2, 'LAI');
 

// Display
Map.addLayer(sentinel2.select('LAI'), {min: 0, max: 5}, 'LAI');
//Map.addLayer(sentinel2.select('B1'), {min: 0, max: 5}, 'B1');


