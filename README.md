# Microsoft Capstone: Huckleberry Habitat & (MAYBE) Yield Prediction

## Project Overview
This project aims to analyze and predict huckleberry habitat suitability and potential yield using environmental data and machine learning techniques. The analysis combines historical occurrence data with climate and environmental variables to understand the factors that influence huckleberry growth and distribution.

## Project Structure
```
├── data/
│   ├── raw/           # Original, immutable data
│   └── processed/     # Cleaned and processed data
├── notebooks/         # Jupyter notebooks for analysis
├── docs/             # Project documentation
└── src/              # Source code
```

## Data Sources
[GBIF.org (11 June 2025) GBIF Occurrence Download]("https://doi.org/10.15468/dl.jzaue9")  

[Why Are Huckleberries Not Grown Commercially?](https://shuncy.com/article/why-are-huckleberries-not-grown-commercially)

[Huckleberry Monitoring in the Gifford Pinchot National Forest](https://www.cascadeforest.org/wp-content/uploads/2021/09/2019-CFC-Huckleberry-Report-2.pdf?utm_source=chatgpt.com)

[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/gridmet)

[Xarray Tutorial Notebook](https://tutorial.xarray.dev/overview/xarray-in-45-min.html)

## Project Goals
1. Analyze historical huckleberry occurrence data to identify key habitat characteristics
2. Integrate climate data from GridMET to understand environmental influences
3. Develop predictive models for habitat suitability and potential yield
4. Create visualizations to aid in understanding huckleberry distribution patterns

## Key Features
- Data cleaning and preparation of GBIF occurrence records
- Integration of environmental variables from GridMET
- Machine learning models for habitat prediction
- Geospatial analysis and visualization
- Yield prediction based on environmental factors

## Getting Started
1. Clone the repository
2. Install required dependencies
3. Follow the notebooks in order to reproduce the analysis

## Dependencies
- Python 3.x
- pandas
- geopandas
- xarray
- scikit-learn
- Additional dependencies listed in requirements.txt

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

