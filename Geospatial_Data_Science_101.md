# Table of Contents

1. **Introduction to Geospatial Data Science**

   - What is Geospatial Data Science?
   - Why is Geospatial Data Science Important?
   - Key Concepts in Geospatial Data Science
   - Getting Started with Python for Geospatial Data

2. **Understanding Geospatial Data**

   - Types of geospatial data
   - Common geospatial file formats
   - Sources of geospatial data
   - Acquiring geospatial data
 
3. **Spatial Data Visualization**

   - The importance of geospatial visualization
   - Types of spatial visualizations
   - Python libraries for spatial data visualization 

4. **Spatial Data Analysis**

   - Spatial Joins
   - Spatial Statistics
   - Spatial Clustering
   - Spatial Interpolation
   - Kernel Density Estimation (KDE)

5. **Spatial Machine Learning**

   - Key Concepts in Spatial Machine Learning
   - Spatial Regression
   - Spatial Classification
   - Spatial Clustering
   - Deep Learning for Spatial Data
   - Feature Engineering for Spatial Data
   - Model Evaluation for Spatial Data


6. **Real-World Geospatial Case Studies**

   - Environmental Monitoring: Air Pollution Prediction
   - Urban Planning: Optimal Site Selection for Hospitals
   - Transportation: Traffic Congestion Analysis
   - Agriculture: Crop Yield Prediction
   - Disaster Management: Flood Risk Mapping

7. **Apache Sedona**
   - Key Features of Apache Sedona
   - Installing Apache Sedona
   - Loading and Processing Geospatial Data
   - Performing Spatial Operations
   - Spatial Indexing for Performance Optimization
   - Visualizing Geospatial Data

8. **Building a Geospatial Data Science Project**
   - Defining a Problem and Collecting Data
   - Processing and Analyzing Data
   - Presenting Results and Insights

9. **Next Steps and Furhter Learning**
   - Recommended Books and Courses
   - Communities and Online Resources

---

# Chapter 1: Introduction to Geospatial Data Science 

## What is Geospatial Data Science?
Geospatial Data Science is an interdisciplinary field that integrates spatial data with data science techniques to derive meaningful insights. It involves working with georeferenced data, performing spatial analysis, applying machine learning models, and leveraging Geographic Information Systems (GIS) to understand spatial relationships and patterns. The field is at the intersection of geography, data science, and computer science, enabling professionals to analyze and visualize spatial data efficiently.

## Why is Geospatial Data Science Important? 
Geospatial data is fundamental to decision-making across numerous industries. Understanding spatial relationships can help optimize resources, improve efficiency, and drive strategic planning. Below are some key areas where geospatial data science plays a vital role:

- **Urban Planning**: Analyzing population density, land use, and transportation networks to support sustainable city development.
- **Environmental Monitoring**: Assessing the impact of climate change, deforestation, pollution levels, and natural disasters.
- **Healthcare & Epidemiology**: Mapping disease outbreaks, tracking the spread of pandemics, and optimizing healthcare facility locations.
- **Agriculture**: Analyzing soil health, crop yield predictions, and irrigation management using satellite imagery.
- **Transportation & Logistics**: Optimizing delivery routes, analyzing traffic congestion, and improving public transportation systems.
- **Retail & Marketing**: Identifying high-potential market locations, optimizing supply chain distribution, and targeting customers based on geolocation data.

## Key Concepts in Geospatial Data Science
Understanding the fundamental concepts of geospatial data science is crucial for working effectively with spatial data. Below are the core concepts:

1. **Spatial Data**: Data that contains location-based information, typically represented as latitude and longitude coordinates.
2. **Coordinate Reference System (CRS)**: A system used to define how spatial data is projected onto a two-dimensional plane for accurate mapping.
3. **Vector Data**: Represented as points (e.g., city locations), lines (e.g., roads, rivers), and polygons (e.g., country boundaries, land parcels).
4. **Raster Data**: Grid-based data, often derived from satellite imagery and elevation models, where each cell in the grid contains a value representing a particular attribute (e.g., temperature, elevation).
5. **Geospatial Analysis**: Methods used to study spatial relationships, including spatial clustering, spatial interpolation, and network analysis.
6. **Geocoding & Reverse Geocoding**: The process of converting addresses into geographic coordinates (geocoding) and vice versa (reverse geocoding).

## Getting Started with Python for Geospatial Data
Python is widely used in geospatial data science due to its extensive ecosystem of libraries and ease of integration with GIS platforms. Some essential Python libraries for geospatial data science include:

- **GeoPandas**: Extends Pandas to handle spatial data efficiently.
- **Shapely**: Provides functions for geometric operations, such as calculating distances and intersections.
- **Fiona**: Reads and writes spatial data formats like Shapefiles and GeoJSON.
- **Rasterio**: Processes and analyzes raster datasets, such as satellite imagery.
- **Folium**: Creates interactive web-based maps using Leaflet.js.
- **Geopy**: Performs geocoding, reverse geocoding, and location-based distance calculations.

### Python Code Example: Basic Geospatial Libraries
```python
# Install necessary libraries (if not installed)
!pip install geopandas rasterio folium shapely geopy

# Import geospatial libraries
import geopandas as gpd
import rasterio
import folium
from shapely.geometry import Point
from geopy.geocoders import Nominatim

# Check library versions
print("GeoPandas version:", gpd.__version__)
print("Rasterio version:", rasterio.__version__)
print("Folium version:", folium.__version__)

# Create a simple point geometry
point = Point(106.8456, -6.2088)  # Jakarta coordinates
print("Example Point:", point)

# Create a simple map using Folium
m = folium.Map(location=[-6.2088, 106.8456], zoom_start=12)
folium.Marker([-6.2088, 106.8456], popup="Jakarta").add_to(m)
m

```

## Summary
In this chapter, we introduced the fundamentals of geospatial data science, its importance, and key concepts such as spatial data types and analysis techniques. We also explored the key Python libraries for working with geospatial data and provided an example of using Python to perform basic geospatial operations. 

In the next chapter, we will dive deeper into different types of geospatial data and their characteristics, including how to acquire and preprocess geospatial datasets.

# Chapter 2: Understanding Geospatial Data

Geospatial data is fundamental to geospatial data science. It represents information associated with specific locations on the Earth's surface. Understanding different types of geospatial data, their sources, formats, and how to process them is crucial for anyone aiming to work in this field. This chapter covers:

- The types of geospatial data
- Common geospatial file formats
- Sources of geospatial data
- Methods to acquire and preprocess geospatial data
- Working with geospatial data in Python

## Types of Geospatial Data
Geospatial data is primarily classified into two categories:

### 1. Vector Data
Vector data represents geographic features using geometric shapes:
- **Point Data**: Represents specific locations (e.g., city locations, ATMs, weather stations).
- **Line Data**: Represents linear features (e.g., roads, rivers, railways).
- **Polygon Data**: Represents area-based features (e.g., country borders, land use zones, lakes).

### 2. Raster Data
Raster data consists of grid-based pixels where each cell contains a value representing an attribute (e.g., elevation, temperature, satellite imagery). Raster data is commonly used for:
- **Satellite Imagery**: Captures Earth's surface for environmental monitoring.
- **Digital Elevation Models (DEM)**: Represents terrain elevation.
- **Aerial Photography**: Used for land use analysis.

## Common Geospatial File Formats
### 1. **Shapefile (.shp)**
Shapefiles are widely used for storing vector data. They consist of multiple files:
- `.shp` - Stores geometry.
- `.shx` - Spatial index for quick access.
- `.dbf` - Attribute data (table format).
- `.prj` - Projection information (coordinate reference system).

To read a shapefile using Python:
```python
import geopandas as gpd

# Load shapefile
shapefile_path = "data/your_shapefile.shp"
gdf = gpd.read_file(shapefile_path)
print(gdf.head())
```

### 2. **Keyhole Markup Language (KML) & Keyhole Markup Zip (KMZ)**
- **KML**: A file format used for geographic data visualization, mainly in Google Earth.
- **KMZ**: A compressed version of KML.

To read a KML file using Python:
```python
import geopandas as gpd

# Load KML file
kml_path = "data/your_file.kml"
gdf = gpd.read_file(kml_path)
print(gdf.head())
```

### 3. **GeoJSON**
GeoJSON is a widely used format for encoding geographic data structures using JavaScript Object Notation (JSON). It supports different types of geometries, such as:
- **Point**: Represents a single location (longitude, latitude).
- **LineString**: Represents a sequence of points forming a line.
- **Polygon**: Represents an enclosed area with multiple points.
- **MultiPoint, MultiLineString, MultiPolygon**: Variants for multiple geometries.
- **Feature & FeatureCollection**: Used to store multiple geometric objects with associated properties.

Example of a simple GeoJSON file:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-122.4194, 37.7749]
      },
      "properties": {
        "name": "San Francisco"
      }
    }
  ]
}
```

#### Loading GeoJSON in Python using GeoPandas
```python
import geopandas as gpd
# Load a GeoJSON file
geo_df = gpd.read_file("data.geojson")
print(geo_df.head())
```

## Sources of Geospatial Data
You can obtain geospatial data from various sources:
- **OpenStreetMap (OSM)**: Open-source vector data.
- **USGS Earth Explorer**: Satellite imagery and elevation data.
- **Google Earth Engine**: Cloud-based platform for large-scale analysis.
- **Government Portals**: Many national agencies provide open geospatial datasets.

## Acquiring Geospatial Data Using Python
Python provides powerful libraries to download and analyze geospatial data.

### Example: Downloading OpenStreetMap (OSM) Data
```python
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

# Define a location
place_name = "Jakarta, Indonesia"

# Download street network
graph = ox.graph_from_place(place_name, network_type='drive')

# Convert graph to GeoDataFrame
gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)

# Plot the street network
fig, ax = plt.subplots(figsize=(10, 10))
gdf_edges.plot(ax=ax, linewidth=0.5, color='blue')
plt.title("Street Network of Jakarta")
plt.show()
```

### Converting CSV Files to Geospatial Data
If you have a CSV file containing location data, you can convert it to a GeoDataFrame.

#### 1. **When the CSV has a Geometry Column**
Some CSV files contain a geometry column with `POINT(x y)` format.
```python
import geopandas as gpd

# Load CSV file with a geometry column
csv_path = "data/your_data.csv"
gdf = gpd.read_file(csv_path)
print(gdf.head())
```

#### 2. **When the CSV has Latitude and Longitude Columns**
If latitude and longitude are stored as separate columns, you can create a geometry column.
```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Load CSV file
csv_path = "data/your_data.csv"
df = pd.read_csv(csv_path)

# Convert latitude and longitude into a geometry column
df["geometry"] = df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)

# Convert DataFrame to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
print(gdf.head())
```

## Summary
In this chapter, we explored the types of geospatial data, common file formats, data sources, and preprocessing techniques using Python. We also demonstrated how to convert CSV files to GeoDataFrames. In the next chapter, we will discuss spatial data visualization techniques to effectively represent geospatial information.


# Chapter 3: Spatial Data Visualization

Visualization is a crucial aspect of geospatial data science, as it helps in understanding spatial relationships, patterns, and trends. Effective spatial visualizations allow analysts to interpret complex geospatial datasets and make data-driven decisions.

This chapter will cover:
- The importance of geospatial visualization
- Common types of spatial visualizations
- Python libraries for spatial data visualization
- Practical examples using `geopandas`, `folium`, and `matplotlib`

## The Importance of Geospatial Visualization
Geospatial visualizations help to:
- Identify patterns and trends in spatial data
- Represent geographic distributions effectively
- Support decision-making with insightful spatial representations
- Improve communication of spatial findings to stakeholders

## Types of Spatial Visualizations
### 1. Choropleth Maps
Choropleth maps use different shades or colors to represent varying values across different geographic areas.

### 2. Heatmaps
Heatmaps represent data density over a geographical area using color intensity, highlighting areas with higher data concentration.

### 3. Point Maps
Point maps plot individual points (e.g., locations of businesses, crime incidents, or weather stations) on a map.

### 4. Line Maps
Line maps represent linear features such as roads, rivers, and transport networks.

### 5. Polygon Maps
Polygon maps display area-based data, such as country boundaries or land-use zones.

### 6. Interactive Maps
Interactive maps allow users to zoom, pan, and explore data dynamically, enhancing user engagement and insight discovery.

## Python Libraries for Geospatial Visualization
Several Python libraries are commonly used for spatial visualization:
- **`geopandas`**: Used for static visualization of vector data.
- **`matplotlib` & `seaborn`**: Standard plotting libraries for static spatial representations.
- **`folium`**: Creates interactive web-based maps.
- **`plotly`**: Provides interactive geospatial plotting capabilities.
- **`kepler.gl`**: Used for advanced geospatial visualization in Jupyter notebooks.

### Example 1: Plotting a Choropleth Map with `geopandas`
```python
import geopandas as gpd
import matplotlib.pyplot as plt

# Load a shapefile of world countries
gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plot a choropleth map of world population
gdf.plot(column='pop_est', cmap='OrRd', legend=True, figsize=(10,6))
plt.title("World Population Map")
plt.show()
```

### Example 2: Creating an Interactive Map with `folium`
```python
import folium

# Define map center and zoom level
m = folium.Map(location=[-6.2088, 106.8456], zoom_start=10)

# Add a marker for Jakarta
folium.Marker(
    location=[-6.2088, 106.8456],
    popup='Jakarta, Indonesia',
    icon=folium.Icon(color='blue')
).add_to(m)

# Save and display map
m.save("interactive_map.html")
print("Interactive map saved as interactive_map.html")
```

### Example 3: Creating a Heatmap with `folium.plugins`
```python
from folium.plugins import HeatMap
import pandas as pd

# Sample data (latitude, longitude, intensity)
data = [
    [-6.2088, 106.8456, 5],  # Jakarta
    [-7.2504, 112.7688, 3],  # Surabaya
    [-6.9147, 107.6098, 4]   # Bandung
]

# Create a folium map
m = folium.Map(location=[-6.5, 107], zoom_start=6)
HeatMap(data).add_to(m)
m.save("heatmap.html")
print("Heatmap saved as heatmap.html")
```

### Example 4: Plotting Point Data on a Static Map
```python
# Load point dataset (e.g., city locations)
cities = gpd.GeoDataFrame({
    'city': ['Jakarta', 'Surabaya', 'Bandung'],
    'geometry': gpd.points_from_xy([106.8456, 112.7688, 107.6098], [-6.2088, -7.2504, -6.9147])
})

# Plot the points on a map
fig, ax = plt.subplots(figsize=(8,6))
gdf.plot(ax=ax, color='lightgrey')  # Base map
cities.plot(ax=ax, color='red', markersize=50)
plt.title("Major Cities in Indonesia")
plt.show()
```

## Summary
In this chapter, we covered:
- The importance of geospatial visualization
- Different types of spatial visualizations
- Key Python libraries for geospatial mapping
- Practical examples of spatial visualization using `geopandas`, `folium`, and `matplotlib`

In the next chapter, we will explore spatial data analysis techniques, including spatial joins, spatial statistics, and spatial clustering.


# Chapter 4: Spatial Data Analysis

Spatial data analysis involves examining geographic data to identify patterns, relationships, and trends. This chapter covers essential spatial analysis techniques, including spatial joins, spatial statistics, and spatial clustering. We will also provide Python examples to demonstrate these concepts.

1. **Spatial Joins**: Combining attributes from one spatial dataset with another based on their spatial relationship.
2. **Spatial Statistics**: Using statistical methods to analyze spatial distributions and relationships.
3. **Spatial Clustering**: Identifying clusters or hotspots within spatial data.
4. **Spatial Interpolation**: Estimating unknown values at certain locations based on known values from surrounding points.
5. **Kernel Density Estimation (KDE)**: Analyzing point density to identify areas of high concentration.

## 1. Spatial Joins
Spatial joins allow merging of two spatial datasets based on their location. 

### Example: Joining Point Data to Polygon Data
```python
import geopandas as gpd

# Load polygon dataset (e.g., administrative boundaries)
districts = gpd.read_file("districts.shp")

# Load point dataset (e.g., store locations)
stores = gpd.read_file("stores.shp")

# Perform spatial join
stores_in_districts = gpd.sjoin(stores, districts, how="left", predicate="intersects")
print(stores_in_districts.head())
```

### Example: Distance-Based Spatial Join
```python
# Perform a spatial join based on distance
stores_near_districts = gpd.sjoin_nearest(stores, districts, how="left")
print(stores_near_districts.head())
```

## 2. Spatial Statistics
Spatial statistics help measure spatial relationships and distributions.

### Example: Measuring Spatial Autocorrelation (Moran’s I)
```python
import esda
import libpysal
from esda.moran import Moran
import numpy as np

# Load spatial data
gdf = gpd.read_file("spatial_data.shp")

# Create spatial weights matrix
w = libpysal.weights.Queen.from_dataframe(gdf)
w.transform = "r"

# Compute Moran's I
moran = Moran(gdf["value_column"], w)
print("Moran’s I value:", moran.I)
```

### Example: Local Indicators of Spatial Association (LISA)
```python
from esda.moran import Moran_Local

# Compute LISA statistic
lisa = Moran_Local(gdf["value_column"], w)
print("LISA values:", lisa.Is)
```

## 3. Spatial Clustering
Spatial clustering identifies patterns and groups within spatial datasets.

### Example: DBSCAN Clustering
```python
from sklearn.cluster import DBSCAN
import numpy as np

# Convert geometries to coordinates
coords = np.array([(point.x, point.y) for point in gdf.geometry])

# Apply DBSCAN clustering
clustering = DBSCAN(eps=0.05, min_samples=5).fit(coords)
gdf["cluster"] = clustering.labels_
print(gdf.head())
```

### Example: K-Means Clustering for Spatial Data
```python
from sklearn.cluster import KMeans

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(coords)
gdf["cluster"] = kmeans.labels_
print(gdf.head())
```

## 4. Spatial Interpolation
Spatial interpolation estimates missing values at certain locations using known data points.

### Example: IDW Interpolation
```python
from scipy.interpolate import Rbf

# Create an IDW model
rbf = Rbf(gdf.geometry.x, gdf.geometry.y, gdf["value_column"], function='linear')

# Predict value at a new location
new_value = rbf(100.5, 200.3)
print("Predicted value:", new_value)
```

## 5. Kernel Density Estimation (KDE)
KDE is useful for visualizing point density distributions.

### Example: KDE for Point Data
```python
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Extract point coordinates
coords = np.array([(point.x, point.y) for point in gdf.geometry])

# Plot KDE using Seaborn
sns.kdeplot(x=coords[:, 0], y=coords[:, 1], cmap='Reds', fill=True)
plt.show()
```

## Summary
In this chapter, we covered:
- **Spatial Joins** for merging spatial datasets
- **Spatial Statistics** like Moran’s I and LISA for spatial autocorrelation
- **Spatial Clustering** techniques such as DBSCAN and K-Means
- **Spatial Interpolation** for predicting unknown values
- **Kernel Density Estimation (KDE)** for analyzing point density

Next, we will explore spatial machine learning techniques in geospatial data science.


# Chapter 5: Spatial Machine Learning

Spatial Machine Learning applies traditional and advanced machine learning techniques to geographic data. Unlike standard machine learning, spatial machine learning considers spatial dependencies, relationships, and heterogeneity in the data. This chapter explores key concepts, methods, and applications of machine learning in geospatial contexts, covering regression, classification, clustering, and deep learning approaches. We will also discuss feature engineering techniques specific to spatial data and evaluation metrics for spatial models.



## 1. Key Concepts in Spatial Machine Learning
1. **Spatial Dependency**: Nearby locations often influence each other (Tobler’s First Law of Geography).
2. **Spatial Heterogeneity**: The relationships between variables may vary across different geographic regions.
3. **Spatial Autocorrelation**: Measures the degree of similarity between spatially close observations.
4. **Feature Engineering for Spatial Data**: Extracting meaningful spatial features to enhance predictive models.
5. **Evaluation Metrics for Spatial Models**: Traditional metrics (e.g., RMSE, accuracy) combined with spatial diagnostics (e.g., Moran’s I).

---

## 2. Spatial Regression
Spatial regression is used to predict continuous values based on spatial features, such as predicting real estate prices, air pollution levels, or traffic congestion.

### Example: Geographically Weighted Regression (GWR)
```python
import geopandas as gpd
import numpy as np
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# Load spatial dataset
gdf = gpd.read_file("real_estate.shp")

# Define independent and dependent variables
y = gdf["price"].values.reshape(-1, 1)  # Target variable
X = gdf[["population", "income"]].values  # Features

# Extract coordinates
coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))

# Select optimal bandwidth
bw = Sel_BW(coords, y, X).search()

# Fit GWR model
gwr_model = GWR(coords, y, X, bw).fit()
print(gwr_model.summary())
```

---

## 3. Spatial Classification
Spatial classification is used to categorize geographic entities, such as land cover types, crime risk zones, or customer segmentation.

### Example: Land Cover Classification using Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
gdf = gpd.read_file("landcover_data.shp")

# Prepare features and labels
X = gdf[["NDVI", "elevation", "slope"]]
y = gdf["land_cover_type"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
predictions = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

---

## 4. Spatial Clustering
Spatial clustering is used to detect patterns and group similar geographic data points.

### Example: Identifying High Crime Areas using DBSCAN
```python
from sklearn.cluster import DBSCAN
import numpy as np

# Extract coordinates
coords = np.array([(point.x, point.y) for point in gdf.geometry])

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.01, min_samples=5)
gdf["cluster"] = dbscan.fit_predict(coords)
print(gdf.head())
```

---

## 5. Deep Learning for Spatial Data
Deep learning techniques, such as Convolutional Neural Networks (CNNs), are used for analyzing satellite imagery and geospatial raster data.

### Example: CNN for Satellite Image Classification
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 land cover classes
])

# Compile and summarize model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## 6. Feature Engineering for Spatial Data
Feature engineering is crucial for improving model accuracy by generating meaningful spatial attributes.

### Example: Creating Spatial Features
```python
# Calculate distance to nearest road
roads = gpd.read_file("roads.shp")
gdf["dist_to_road"] = gdf.geometry.apply(lambda x: roads.distance(x).min())

# Extract elevation data from raster
def get_elevation_from_raster(geometry):
    # Simulated function to get elevation from raster
    return np.random.randint(100, 500)

gdf["elevation"] = gdf.geometry.apply(get_elevation_from_raster)

print(gdf.head())
```

---

## 7. Model Evaluation for Spatial Data
Evaluating spatial models requires traditional metrics along with spatial-specific diagnostics.

### Evaluation Metrics:
1. **RMSE (Root Mean Squared Error)**: Measures regression model performance.
2. **Moran’s I**: Measures spatial autocorrelation in model residuals.
3. **Silhouette Score**: Evaluates clustering quality.
4. **Overall Accuracy & Kappa Score**: Assesses classification models.

### Example: Checking Moran’s I for Spatial Autocorrelation
```python
import esda
from libpysal.weights import KNN
from esda.moran import Moran

# Create spatial weight matrix
w = KNN.from_dataframe(gdf, k=5)

# Compute Moran’s I
moran = Moran(gdf["residuals"], w)
print("Moran's I:", moran.I)
```

---

## Summary
In this chapter, we explored:
- **Spatial Regression** using Geographically Weighted Regression (GWR)
- **Spatial Classification** with Random Forest
- **Spatial Clustering** using DBSCAN
- **Deep Learning for Spatial Data** with CNNs
- **Feature Engineering for Spatial Data**
- **Evaluation Metrics for Spatial Models**

The next chapter will focus on real-world geospatial case studies, demonstrating how these techniques can be applied in various industries!


# Chapter 6: Real-World Geospatial Case Studies

This chapter explores real-world applications of geospatial data science in various industries. We will examine case studies in environmental monitoring, urban planning, transportation, agriculture, and disaster management. Each case study includes the problem statement, data sources, methodologies, and Python implementations.


## 1. Environmental Monitoring: Air Pollution Prediction

### Problem Statement
Air pollution is a major public health concern. Predicting pollution levels in different locations helps authorities take proactive measures.

### Data Sources
- **Satellite Imagery** (NASA, Sentinel-5P)
- **Ground-based Air Quality Stations** (Government agencies)
- **Meteorological Data** (Temperature, Wind Speed, Humidity)

### Methodology
- Collect and preprocess air pollution data.
- Extract spatial features (e.g., proximity to highways, land use).
- Train a spatial regression model to predict PM2.5 levels.

### Example: Predicting PM2.5 Concentration
```python
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load air quality data
gdf = gpd.read_file("air_quality.shp")

# Define features and target variable
X = gdf[["temperature", "wind_speed", "humidity", "traffic_density"]]
y = gdf["PM2.5"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict pollution levels
gdf["predicted_PM2.5"] = model.predict(X)
print(gdf.head())
```

---

## 2. Urban Planning: Optimal Site Selection for Hospitals

### Problem Statement
City planners need to determine the best locations for new hospitals based on accessibility and population density.

### Data Sources
- **Demographic Data** (Population distribution)
- **Road Network Data**
- **Existing Healthcare Facility Locations**

### Methodology
- Use GIS tools to identify underserved areas.
- Compute travel time and accessibility.
- Apply spatial optimization to find the best hospital sites.

### Example: Finding Nearest Hospitals
```python
from shapely.ops import nearest_points

# Load data
gdf_hospitals = gpd.read_file("hospitals.shp")
gdf_population = gpd.read_file("population.shp")

# Find nearest hospital for each population center
def find_nearest(row, hospital_gdf):
    nearest_geom = nearest_points(row.geometry, hospital_gdf.unary_union)[1]
    return hospital_gdf[hospital_gdf.geometry == nearest_geom].index[0]

gdf_population["nearest_hospital"] = gdf_population.apply(lambda row: find_nearest(row, gdf_hospitals), axis=1)
print(gdf_population.head())
```

---

## 3. Transportation: Traffic Congestion Analysis

### Problem Statement
Traffic congestion leads to delays and economic losses. Analyzing traffic patterns helps in designing better road networks.

### Data Sources
- **GPS Trajectories** (Taxi, Ride-sharing data)
- **Road Network Data**
- **Traffic Sensor Data**

### Methodology
- Aggregate traffic data by time and location.
- Detect congestion hotspots using clustering.
- Predict congestion trends using machine learning.

### Example: Identifying Congested Areas
```python
from sklearn.cluster import DBSCAN
import numpy as np

# Extract coordinates
coords = np.array([(geom.x, geom.y) for geom in gdf_traffic.geometry])

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.005, min_samples=10)
gdf_traffic["congestion_cluster"] = dbscan.fit_predict(coords)
print(gdf_traffic.head())
```

---

## 4. Agriculture: Crop Yield Prediction

### Problem Statement
Predicting crop yields helps farmers and policymakers make informed decisions about food production and supply chains.

### Data Sources
- **Satellite Imagery** (NDVI, Soil Moisture)
- **Weather Data**
- **Historical Crop Yield Records**

### Methodology
- Preprocess and clean agricultural data.
- Train a machine learning model using NDVI and weather features.
- Predict crop yield for different regions.

### Example: Predicting Wheat Yield
```python
from sklearn.linear_model import LinearRegression

# Load crop yield dataset
df = pd.read_csv("crop_yield.csv")

# Train model
model = LinearRegression()
X = df[["NDVI", "rainfall", "temperature"]]
y = df["yield"]
model.fit(X, y)

# Predict yield
df["predicted_yield"] = model.predict(X)
print(df.head())
```

---

## 5. Disaster Management: Flood Risk Mapping

### Problem Statement
Predicting flood-prone areas enables authorities to take preventive measures and reduce damage.

### Data Sources
- **Topographic Data** (Elevation, Slope)
- **Rainfall Data**
- **Land Use/Land Cover Data**

### Methodology
- Combine elevation, rainfall, and land cover data.
- Generate a flood risk index using spatial analysis.
- Visualize flood-prone areas on a map.

### Example: Flood Risk Analysis
```python
import rasterio
import numpy as np

# Load elevation data
dem = rasterio.open("dem.tif")
elevation = dem.read(1)

# Compute slope
slope = np.gradient(elevation)

# Define flood risk index
flood_risk = (slope < 5) * (rainfall > 200) * (land_cover == "urban")
print("Flood-prone areas identified")
```

---

## Summary
In this chapter, we explored:
- **Air Pollution Prediction** using spatial regression.
- **Hospital Site Selection** using spatial optimization.
- **Traffic Congestion Analysis** with clustering.
- **Crop Yield Prediction** using remote sensing and machine learning.
- **Flood Risk Mapping** using GIS and spatial analysis.

These case studies demonstrate the power of geospatial data science in solving real-world problems. In the next chapter, we will discuss best practices for handling geospatial big data and cloud computing solutions!

# Chapter 7: Apache Sedona

Apache Sedona (formerly known as GeoSpark) is a powerful open-source distributed computing framework for processing large-scale geospatial data. Built on top of Apache Spark, Sedona provides efficient spatial data processing capabilities, making it ideal for handling big geospatial data. In this chapter, we will explore its features, installation, core functionalities, and practical examples.

---

## Key Features of Apache Sedona
- **Distributed Spatial Computing**: Leverages Spark’s distributed computing capabilities to process large geospatial datasets efficiently.
- **Support for Multiple Formats**: Works with Shapefiles, GeoJSON, WKT, WKB, and PostGIS databases.
- **Spatial SQL Support**: Enables users to run spatial queries using Spark SQL.
- **Spatial Indexing**: Supports R-Tree and Quad-Tree indexing for faster query execution.
- **Geospatial Analytics**: Includes spatial join, clustering, and visualization tools.

---

## Installing Apache Sedona
To use Apache Sedona with PySpark, install the necessary libraries:
```bash
pip install apache-sedona
```
Alternatively, when using Apache Sedona with Spark, add the Sedona JAR files to your Spark session:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Apache Sedona Example") \
    .config("spark.jars.packages", "org.apache.sedona:sedona-python-adapter-3.0_2.12:1.2.1") \
    .getOrCreate()
```

---

## Loading and Processing Geospatial Data
Apache Sedona can load geospatial datasets in various formats.

### Loading a Shapefile into a DataFrame
```python
from sedona.spark import SedonaContext
from sedona.register import SedonaRegistrator

sc = SedonaContext.create(spark)
SedonaRegistrator.registerAll(sc)

shapefile_path = "path/to/shapefile"
spatial_df = spark.read.format("geotools").load(shapefile_path)
spatial_df.show()
```

### Loading GeoJSON Data
```python
geojson_path = "path/to/data.geojson"
geojson_df = spark.read.format("json").load(geojson_path)
geojson_df.show()
```

---

## Performing Spatial Operations
Apache Sedona provides powerful spatial functions that can be used in Spark SQL.

### Registering Spatial SQL Functions
```python
SedonaRegistrator.registerAll(spark)
```

### Running Spatial SQL Queries
```python
spatial_df.createOrReplaceTempView("spatial_data")
query = """
SELECT ST_Contains(geom, ST_GeomFromText('POINT(30 10)')) AS contains
FROM spatial_data
"""
result = spark.sql(query)
result.show()
```

### Spatial Join Example
```python
query = """
SELECT a.id, b.id
FROM spatial_table1 a
JOIN spatial_table2 b
ON ST_Intersects(a.geom, b.geom)
"""
join_result = spark.sql(query)
join_result.show()
```

---

## Spatial Indexing for Performance Optimization
Apache Sedona supports spatial indexing to improve query performance.

### Creating an R-Tree Index
```python
from sedona.core.spatialOperator import RangeQuery

index = spatial_df.selectExpr("ST_GeomFromWKT(geom) as geom").rdd
indexed_rdd = RangeQuery.SpatialRangeQuery(index, "ST_Within", True, False)
```

---

## Visualizing Geospatial Data
Apache Sedona allows exporting results for visualization in GIS tools.

### Exporting Data as GeoJSON
```python
spatial_df.write.format("json").save("output.geojson")
```

---

## Summary
In this chapter, we covered:
- Apache Sedona’s capabilities for large-scale geospatial data processing.
- Installation and setup of Sedona with Spark.
- Loading and processing geospatial data in different formats.
- Running spatial SQL queries and performing spatial joins.
- Utilizing spatial indexing for performance optimization.
- Exporting results for GIS visualization.

Apache Sedona is a powerful tool for handling geospatial big data, enabling scalable geospatial analytics. In the next chapter, we will explore real-time geospatial analytics using streaming data sources!


# Chapter 8: Building a Geospatial Data Science Project

Building a geospatial data science project involves several key steps, including defining a problem, collecting relevant data, processing and analyzing the data, and finally presenting meaningful insights. In this chapter, we will walk through a structured approach to developing a geospatial data science project from start to finish.

---

## Defining a Problem and Collecting Data

### Identifying a Geospatial Problem
Before starting a project, it's crucial to define a clear problem statement. Some common geospatial problems include:
- **Urban Planning**: Identifying the best locations for new infrastructure.
- **Environmental Monitoring**: Analyzing deforestation or air pollution.
- **Disaster Management**: Mapping flood-prone areas or tracking wildfires.
- **Transportation Optimization**: Finding the most efficient routes for delivery services.

### Data Sources for Geospatial Projects
Once the problem is defined, the next step is collecting data. Geospatial data can come from various sources:
- **Open Data Portals**: Governments and organizations provide open geospatial datasets (e.g., OpenStreetMap, NASA EarthData, USGS).
- **Satellite and Remote Sensing Data**: Sources like Sentinel, Landsat, and MODIS offer satellite imagery.
- **APIs**: Google Maps API, OpenWeatherMap API, and Mapbox provide real-time geospatial data.
- **Crowdsourced Data**: Platforms like OpenStreetMap allow users to contribute location-based data.
- **Company or Proprietary Data**: Businesses may have internal geospatial datasets, such as customer locations or sales territories.

Example: Downloading OpenStreetMap Data using `osmnx` in Python:
```python
import osmnx as ox
# Get street network for a city
city_graph = ox.graph_from_place("Jakarta, Indonesia", network_type="drive")
ox.plot_graph(city_graph)
```

---

## Processing and Analyzing Data

### Data Cleaning and Preprocessing
Geospatial data often requires preprocessing before analysis. Common tasks include:
- **Handling Missing Values**: Fill or remove missing coordinates.
- **Coordinate Reference System (CRS) Transformation**: Convert data to a common spatial reference system.
- **Filtering and Clipping**: Extract relevant features based on area of interest.

Example: Converting CSV Data with Latitude/Longitude to a GeoDataFrame:
```python
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load CSV with lat/lon columns
df = pd.read_csv("locations.csv")
df["geometry"] = df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
gdf.head()
```

### Performing Geospatial Analysis
After preprocessing, geospatial analysis can be conducted to extract insights. Some key techniques include:
- **Spatial Joins**: Combining datasets based on geographic proximity.
- **Distance Analysis**: Calculating distances between locations.
- **Hotspot Detection**: Identifying areas with high concentrations of activity.
- **Geospatial Clustering**: Grouping similar locations (e.g., k-means clustering).

Example: Performing a Spatial Join to Find Points Within a Polygon:
```python
# Load shapefile of city districts
districts = gpd.read_file("districts.shp")

# Spatial join: Find which district each point belongs to
joined = gpd.sjoin(gdf, districts, how="left", predicate="within")
joined.head()
```

---

## Presenting Results and Insights

### Data Visualization Techniques
Effective visualization helps communicate geospatial findings clearly. Some common techniques include:
- **Choropleth Maps**: Representing data intensity using color gradients.
- **Heatmaps**: Highlighting density of points.
- **Interactive Maps**: Allowing users to explore geospatial data dynamically.

Example: Creating a Choropleth Map with `folium`:
```python
import folium
# Create a map centered at a specific location
m = folium.Map(location=[-6.2088, 106.8456], zoom_start=10)
# Add GeoJSON layer
folium.Choropleth(
    geo_data="districts.geojson",
    name="choropleth",
    data=joined,
    columns=["district_name", "population"],
    key_on="feature.properties.district_name",
    fill_color="YlGn",
    fill_opacity=0.7,
    line_opacity=0.2,
).add_to(m)
# Show map
m
```

### Storytelling with Data
Simply showing maps and charts is not enough; data should be framed as a compelling narrative. To do this:
- Provide context: Explain the problem and why the findings matter.
- Use comparisons: Highlight trends, anomalies, and patterns.
- Offer actionable insights: Recommend next steps based on the results.

Example: Insights from an Urban Planning Project
- "The analysis shows that public transport access is limited in the eastern districts. To improve connectivity, new bus routes should be introduced in these areas."

---

## Summary
In this chapter, we explored:
- Defining a problem and collecting relevant geospatial data.
- Processing and analyzing geospatial data using Python.
- Presenting results effectively through visualization and storytelling.

By following these steps, you can build a structured and impactful geospatial data science project. In the next chapter, we will explore advanced geospatial machine learning techniques!


# Chapter 9: Next Steps and Further Learning

## Recommended Books and Courses
To deepen your knowledge in geospatial data science, consider these resources:

### Books
- **"Geospatial Analysis: A Comprehensive Guide"** by Michael J. de Smith
- **"Python Geospatial Development"** by Erik Westra
- **"Mastering Geospatial Analysis with Python"** by Paul Crickard

### Online Courses
- **"Geospatial and Environmental Analysis"** – Coursera (University of California, Davis)
- **"Python for Geospatial Analysis"** – Udemy
- **"Spatial Data Science"** – DataCamp

---

## Communities and Online Resources
Engaging with geospatial communities can accelerate your learning and career growth:

### Online Communities
- **GIS Stack Exchange** – [https://gis.stackexchange.com/](https://gis.stackexchange.com/)
- **OpenStreetMap Community** – [https://community.openstreetmap.org/](https://community.openstreetmap.org/)
- **Reddit GIS** – [https://www.reddit.com/r/gis/](https://www.reddit.com/r/gis/)

### Geospatial Libraries and Documentation
- **GeoPandas** – [https://geopandas.org/](https://geopandas.org/)
- **Folium** – [https://python-visualization.github.io/folium/](https://python-visualization.github.io/folium/)
- **Rasterio** – [https://rasterio.readthedocs.io/](https://rasterio.readthedocs.io/)

---
