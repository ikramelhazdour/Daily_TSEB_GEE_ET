
# **A GEE TSEB Workflow for Daily 30-m Evapotranspiration (ET)**
## **Overview**

These algorithms constitute the workflow of a Google Earth Engine (GEE) implementation of the Two Source Energy Balance (TSEB) Model. This implementation uses an Artificial Neural Network to retrieve Leaf Area Index (LAI), a simple regression relationship with NDVI to retrieve Canopy Height, and a gap-filling approach based on reference ET and Kc to produce daily Evapotranspiration.

## **Parts of the workflow**
- The first part of the workflow (TSEB_Workflow_ERA5_Forcing) involves preparing the model's inputs through a series of data retrieval and processing steps. This process generates inputs as GEE assets, which are subsequently transformed into outputs, also as GEE assets.
- The second part of the workflow applies a gap-filling approach to the outputs (Instantaneous ET) using the "Gap-filling" algorithm, which can be run directly in the GEE coding environment. The result is a daily Evapotranspiration product at a 30-meter resolution.

  ## **Requirements**
- The users are invited to download the geeet package: https://github.com/kaust-halo/geeet, and replace the TSEB module with the tseb.py module.

## **Authors (software) & Aknowledgment**
Ikram EL HAZDOUR (PhD candidate at CESBIO and UCAM)
Michel LE PAGE (Researcher at CESBIO IRD)
Oliver LOPEZ (Researcher at KAUST)
The algorithms are made available for research purposes. Any other use, including commercial applications, requires permission from the author.
Please use the citation below when referencing any research that uses these algorithms.

## **Citation**
The research paper related to this workflow is accessible at : https://doi.org/10.1016/j.envsoft.2025.106365


