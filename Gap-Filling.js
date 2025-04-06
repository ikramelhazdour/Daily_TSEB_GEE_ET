
//========================================================================
//Authors : Ikram EL HAZDOUR & Michel LE PAGE
//========================================================================
/*
This algorithm calculates ET0 using the Penman-Monteith equation and ERA5 data. 
It then combines the ET0 product with the exported TSEB ETa,
applying temporal interpolation to generate daily ETa values. 
The resulting ETa data can be exported by region into a CSV file.
*/
//=====================================================================
// Define date range
var startDate = '2016-01-01';
var endDate = '2023-12-31';

// Import regions (upload your personal shapefile as an asset on GEE)
var regions = ee.FeatureCollection('projects/ee-ikramelhazdour/assets/parcelles_chichaoua')
                                 .filter(ee.Filter.inList('id', [1, 2, 3, 4, 5]));

// Load the TSEB Outputs Collection directly

var TSEB_Collection = ee.ImageCollection('projects/ee-ikramelhazdour/assets/TSEB_outputs_04032024_6')
                                         .filterDate(startDate, endDate).filterBounds(regions);

TSEB_Collection = TSEB_Collection.map(function(image) {
  var dateAcquired = image.get('DATE_ACQUIRED');
  var date = ee.Date.parse('YYYY-MM-dd', dateAcquired);
  return image.set('system:time_start', date.millis());
});


//*************************************************************************************
// Load ET0 Collection (IF ALREADY  EXPORTED)
var ET0 = ee.ImageCollection('projects/ee-ikramelhazdour/assets/era5_et0')
                    .filterDate(startDate, endDate).filterBounds(regions);
//*************************************************************************************

//          Alternative : Compute ET0 using the following algorithm

//*************************************************************************************
//=================================================================
//                   START of The ET0 Caluclation
//=================================================================

/******************************************************
This part of the code computes reference ET (ET0) based 
on the Penman-Monteith Equation and ERA5-Land data
*/
//*****************************************************
var dataset = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate('2016-01-01','2023-12-31');
print(dataset);

// Elevation raster
var elev = ee.Image("USGS/GMTED2010");
// Compute ancillary layers
var lat = ee.Image.pixelLonLat().select('latitude').multiply(Math.PI / 180);
var lon = ee.Image.pixelLonLat().select('longitude').multiply(Math.PI / 180);
var pair = elev.expression('101.3 * pow((293 - 0.0065 * b()) / 293, 5.26)');

// Compute daily ERA5 ETo
function daily_eto_func(fc){
  var date_start = ee.Algorithms.Date(fc.get("system:time_start"));
  var year_start = ee.Number(date_start.get('year'));
  var doy_start = ee.Number(date_start.getRelative('day', 'year')).add(1);

  // This could probably be combined into a single reduction
  // Tair in C, Uz in m s-1, q in kg kg-1
  // Convert Rs from W m-2 to MJ m-2 day-1
  pair = fc.select('surface_pressure');
  pair =pair.divide(1000); // Pa -> kPa
  
  var tmean = fc.select(['temperature_2m']);
  tmean=tmean.add(-273.15); // K -> C

  var tmax = fc.select(['temperature_2m_max']);
  tmax=tmax.add(-273.15); // K -> C

  var tmin = fc.select(['temperature_2m_min']); 
  tmin=tmin.add(-273.15); // K -> C

  var tdew = fc.select(['dewpoint_temperature_2m']);
  tdew=tdew.add(-273.15); // K -> C

//	var rn = fc.select('surface_net_solar_radiation_sum');
 // rn=rn.divide(1000000); // J -> MJ
    // Net radiation (Eqns 15 and 16)
  
	var rn = (fc.select('surface_solar_radiation_downwards_sum').divide(1000000)).multiply(0.77).subtract(fc.select('surface_net_thermal_radiation_sum').divide(1000000));
  
  //Wind Speed (WS) = √(u2 + v2)
  var uz=(fc.select('u_component_of_wind_10m').multiply(fc.select('u_component_of_wind_10m'))
    .add(fc.select('v_component_of_wind_10m').multiply(fc.select('v_component_of_wind_10m')))).sqrt();

  var zw = 10.0;  // Windspeed measurement/estimated height

  var eto = daily_refet_func(doy_start, tmean, tmin, tmax, tdew, rn, uz, zw, 900, 0.34);

  return eto
    .select([0], ['ETo'])
    .set(
      {
        'system:time_start': fc.get('system:time_start'),
        'system:time_end': fc.get('system:time_end')
      });
}

// Vapor Pressure in kPa with temperature in C
function vapor_pressure_func (t) {
  return t.expression('0.6108 * exp(17.27 * b() / (b() + 237.3))')}//

// Daily Reference ET
function daily_refet_func (doy, tmean, tmin, tmax, tdew, rn, uz, zw, cn, cd) {
	//var tmean = tmin.add(tmax).multiply(0.5);  // C
  // To match standardized form, psy is calculated from elevation based pair
	var psy = pair.multiply(0.000665);
  var es_tmax = vapor_pressure_func(tmax);  // C
  var es_tmin = vapor_pressure_func(tmin);  // C
  var es_tmean = vapor_pressure_func(tmean);
  var es_slope = es_tmean.expression(
    '4098 * es / (pow((t + 237.3), 2))', {'es':es_tmean, 't':tmean});
	var es = es_tmin.add(es_tmax).multiply(0.5);
	var ea = vapor_pressure_func(tdew);

  // Wind speed 
	var u2 = uz.expression('b() * 4.87 / log(67.8 * zw - 5.42)', {'zw':zw});

  // Daily reference ET
	return tmin.expression(
		('(0.408 * slope * rn + (psy * cn * u2 * (es - ea) / (t + 273))) / '+
     '(slope + psy * (cd * u2 + 1))'),
		{'slope':es_slope, 'rn':rn, 'psy':psy, 'cn':cn,
     't':tmean, 'u2':u2, 'es':es, 'ea':ea, 'cd':cd});
}

// Calculate daily ETo sum for the collection
var eto_coll = dataset.map(daily_eto_func);


//=================================================================
//                   END of The ET0 Caluclation
//=================================================================

// Join the TSEB and ET0 collections by date
var filterDate = ee.Filter.equals({
    leftField: 'system:time_start',
    rightField: 'system:time_start'
});

var joinedData = ee.Join.inner().apply(TSEB_Collection, ET0, filterDate);
print(joinedData,'joinedData');
joinedData = ee.ImageCollection(joinedData);
print(joinedData,'joinedDataCol');

// Compute Kc using the joined data
var computeKc = function(pair) {
    var imgSSEBOP = ee.Image(pair.get('primary'));
    var imgET0 = ee.Image(pair.get('secondary'));
    var Kc = imgSSEBOP.divide(imgET0);
    return Kc.set('system:time_start', imgSSEBOP.get('system:time_start'));
};

var kcCollection = ee.ImageCollection(joinedData.map(computeKc));
print(kcCollection,'kcCollection');
//====================================================================
//                          INTERPOLATION
//====================================================================
// Create an empty + continuous image collection for interpolation
var bandNames = kcCollection.first().bandNames();
var numBands = bandNames.size();
var initBands = ee.List.repeat(ee.Image(), numBands);
var initImage = ee.ImageCollection(initBands).toBands().rename(bandNames);

// Select the interval (1 image every n days)
var n = 1;
var firstImage = kcCollection.sort('system:time_start').first();
var lastImage = kcCollection.sort('system:time_start', false).first();
var timeStart = ee.Date(firstImage.get('system:time_start'));
var timeEnd = ee.Date(lastImage.get('system:time_start'));
var totalDays = timeEnd.difference(timeStart, 'day');
var daysToInterpolate = ee.List.sequence(0, totalDays, n);

var initImages = daysToInterpolate.map(function(day) {
  return initImage.set({
    'system:index': ee.Number(day).format('%d'),
    'system:time_start': timeStart.advance(day, 'day').millis(),
    'type': 'interpolated'
  });
});

var initCol = ee.ImageCollection.fromImages(initImages);

// Merge original and empty collections for interpolation
var originalCollection = kcCollection.merge(initCol);

// Add timestamp band to each image in the collection
originalCollection = originalCollection.map(function(image) {
  var timeImage = image.metadata('system:time_start').rename('timestamp');
  var timeImageMasked = timeImage.updateMask(image.mask().select(0));
  return image.addBands(timeImageMasked).toFloat();
});

// Define filters for joins
var days = 60; 
var millis = ee.Number(days).multiply(1000*60*60*24);
var maxDiffFilter = ee.Filter.maxDifference({
  difference: millis,
  leftField: 'system:time_start',
  rightField: 'system:time_start'
});
var lessEqFilter = ee.Filter.lessThanOrEquals({
  leftField: 'system:time_start',
  rightField: 'system:time_start'
});
var greaterEqFilter = ee.Filter.greaterThanOrEquals({
  leftField: 'system:time_start',
  rightField: 'system:time_start'
});

// Apply joins for before and after images
var join1 = ee.Join.saveAll({
  matchesKey: 'after',
  ordering: 'system:time_start',
  ascending: false
});
var join1Result = join1.apply({
  primary: originalCollection,
  secondary: originalCollection,
  condition: ee.Filter.and(maxDiffFilter, lessEqFilter)
});
var join2 = ee.Join.saveAll({
  matchesKey: 'before',
  ordering: 'system:time_start',
  ascending: true
});
var join2Result = join2.apply({
  primary: join1Result,
  secondary: join1Result,
  condition: ee.Filter.and(maxDiffFilter, greaterEqFilter)
});
var joinedCol = join2Result;

// Interpolate images
var interpolateImages = function(image) {
  var image = ee.Image(image);
  var beforeImages = ee.List(image.get('before'));
  var beforeMosaic = ee.ImageCollection.fromImages(beforeImages).mosaic();
  var afterImages = ee.List(image.get('after'));
  var afterMosaic = ee.ImageCollection.fromImages(afterImages).mosaic();
  var t1 = beforeMosaic.select('timestamp').rename('t1');
  var t2 = afterMosaic.select('timestamp').rename('t2');
  var t = image.metadata('system:time_start').rename('t');
  var timeImage = ee.Image.cat([t1, t2, t]);
  var timeRatio = timeImage.expression('(t - t1) / (t2 - t1)', {
    't': timeImage.select('t'),
    't1': timeImage.select('t1'),
    't2': timeImage.select('t2'),
  });
  var interpolated = beforeMosaic.add((afterMosaic.subtract(beforeMosaic).multiply(timeRatio)));
  var result = image.unmask(interpolated);
  return result.copyProperties(image, ['system:time_start']);
};

var regularCol = ee.ImageCollection(joinedCol.map(interpolateImages));
regularCol = regularCol.filter(ee.Filter.eq('type', 'interpolated'));

// Compute daily ETA using interpolated Kc and ET0
var computeDailyEta = function(pair) {
   var KcImage = ee.Image(pair.get('primary')).select('ETd');
   var ET0Image = ee.Image(pair.get('secondary')).select('ETo');
   var daily_ETA = ET0Image.multiply(KcImage);
   return daily_ETA.set('system:time_start', KcImage.get('system:time_start'));
};


var joinedEtaData = ee.Join.inner().apply(regularCol, ET0, filterDate);

var ETA = ee.ImageCollection(joinedEtaData.map(computeDailyEta));


//===============================================================
//       EXPORT ETA data for selected parcels in CSV format 
//===============================================================
var name_asset = "doc";
var features=regions.toList(regions.size());

var ETaCollection = ee.ImageCollection(ETA).select('ETo')
  .filterDate(startDate, endDate)
  .filterBounds(regions);


// === Script parameters ===
var st_start = startDate;        // starting date
var st_end = endDate;          // end date
var label_col = "id";              // label column in the shapefile
var folder_out = "/"           // output folder on google drive
var do_ETA = true;               

// reducers: statictics to be computed on the different bands
var reducers = [ee.Reducer.mean(), ee.Reducer.stdDev()]//, ee.Reducer.min(), ee.Reducer.max(), ee.Reducer.count()];


// === Script functions ===
var addArea = function(feature) {
  return feature.set({areaHa: feature.geometry().area().divide(100 * 100)});
};

var addLabel = function(feature) {
  return feature.set({label: feature.get(label_col)});
};

var addTime = function(image) {
  return image.addBands(image.metadata('system:time_start').rename('Date'));
};

function addDate(image) {
  var date = ee.Date(image.get('system:time_start'));
  var dateString = date.format('YYYY-MM-DD');
  return image.set('newDate',dateString);
}


function exportRegions2(table, features,subscript) {
  // EXPORT REGIONS: this version export all the polygons in one file
  // table: the table to export
  // features: the features to export
  // subscript: for example '_ETA'
  // selection: for example ['ETa', 'date','label']  

  Export.table.toDrive({
    collection: table,
    description:"Export"+subscript+"_"+name_asset,
    folder: folder_out,
    fileFormat: 'CSV'
  });
}

//***
//* Combines reducers given array of reducers
//***
function combineReducers(reducers) {
  var current = reducers[0];
  
  reducers.slice(1).forEach(function(r) {
    current = current.combine({ reducer2: r, sharedInputs: true});
  });

  return current;
}

// === Script begins ===

var start = new Date(startDate);
var end = new Date(endDate);

regions= regions.map(addArea);
regions = regions.map(addLabel);

var features=regions.toList(regions.size());
regions = ee.FeatureCollection(features);
var reducer = combineReducers(reducers);

// ---------------------
// EXPORT ETA 
// ---------------------
if (do_ETA === true) {

  var ET_expo =  ETaCollection.filterBounds(regions).select(['ETo']);
  var stats = ET_expo.map(function(img) {
    var dateStr = img.date();
    var dateStr1 = dateStr.format("YYYYMMdd");
    var dateNum = ee.Number.parse(dateStr.format("YYYYMMdd"));
    img = img.addBands(ee.Image(dateNum).rename('date'));
    var reducer = combineReducers(reducers);
    var stats = img.reduceRegions({
      collection: regions,
      reducer: reducer,
      scale: 30
     }).filter(ee.Filter.notNull(['ETo_mean']));
    return(stats);
  }).flatten();

  stats = stats.map(function(ft){
    var dd = ee.Feature(ft).getNumber('date_mean').format("%8d");
    ft = ft.setGeometry(null);  //to reduce the size of the Export table
    return ee.Feature(ft).set('date', dd);
    });
  stats = stats.select(["ETo.*","date","label"]);
  exportRegions2(stats,features,"_ETA");

  }



