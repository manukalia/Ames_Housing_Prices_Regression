___

## Project Two ... Ames Iowa Housing Data Analysis  

<font size='2'>
&nbsp; &nbsp;Manu Kalia Regression Project<br>
&nbsp;&nbsp; 25 - Mar - 2019
</font>
<br><br>  

---

## Kaggle Competition using Ames Iowa Housing Data to Predict Sale Price


### PROBLEM STATEMENT  

<font size='2'>  
Given the provided real estate transaction information in Ames, Iowa…<br>
Can you predict/ explain Sale Price, given the other 80 columns of information (about 2051 rows)?
</font>



### EXECUTIVE SUMMARY  
<br/>
<font size='2'>  

After conducting data cleaning and EDA with some care, some columns were dropped, others were made numerical.  Looked at some preliminary correlations to SalePrice.  Three sets of features were assembled:  set of 73, set of 33, and set of 82 features.  All were used in linear regression pipelines employing standard scaling.  both lass and ridge regularizations were applied.  In one case PolynomialFeatures was tried, as was PowerTransform.  Ultimately, the first attempt, with 73 features and lasso regularization was both a good score and interpretable, and is considered the best solution to this project.  

|Pipeline          |Train Score|Test Score |RMSE     |Interpretation / Comments / Conclusion           |
|---               |---        |---        |---      |---                                              |
|pipe1_lasso       |0.885539   |0.838231   |31471.89 |73 features, LassoCV() regularization, scaled    |
|pipe1_ridge       |0.889490   |0.829837   |32278.14 |73 features, RidgeCV() regularization, scaled    |
|.                 |.          |.          |.        |.                                                |
|pipe2_lasso       |0.884607   |0.836931   |31598.11 |33 features, LassoCV() regularization, scaled    |
|pipe2_ridge       |0.884645   |0.837096   |31582.17 |33 features, RidgeCV() regularization, scaled    |
|.                 |.          |.          |.        |.                                                |
|pipe2_poly_lasso  |0.943194   |0.793612   |35548.20 |33 feat, Poly, Lasso, scaled ... extemely overfit|
|pipe2_poly_ridge  |0.964645   |0.787116   |36103.29 |33 feat, Poly, Lasso, scaled ... extemely overfit|
|.                 |.          |.          |.        |.                                                |
|regr2             |0.859595   |0.669053   |45014.78 |33 feat, PowerTransformed, Lasso regularizzation |
</font>  
<br>




### DATA DICTIONARY

| ITEM and DESCRIPTION                                                                     | TYPE        | ACTION                       |
|------------------------------------------------------------------------------------------|-------------|------------------------------|
|                                                                                          |             |                              |
| •   SalePrice - the property's sale price in dollars.                                    | Numerical   | None                         |
|                                                                                          |             |                              |
| •   MSSubClass: The building class                                                       | Numerical   | None                         |
| •   MSZoning: Identifies the general zoning classification of the sale.                  | Categorical | Future mapping to numerical? |
| •   LotFrontage: Linear feet of street connected to property                             | Numerical   | None                         |
| •   LotArea: Lot size in square feet                                                     | Numerical   | None                         |
| •   Street: Type of road access to property                                              | Categorical | Map to numerical             |
| •   Alley: Type of alley access to property                                              | Categorical | Not Used                     |
| •   LotShape: General shape of property                                                  | Categorical | Map to numerical             |
| •   LandContour: Flatness of the property                                                | Categorical | Map to numerical             |
| •   Utilities: Type of utilities available                                               | Categorical | Map to numerical             |
| •   LotConfig: Lot configuration                                                         | Categorical | Get dummies                  |
| •   LandSlope: Slope of property                                                         | Categorical | Map to numerical             |
| •   Neighborhood: Physical locations within Ames city limits                             | Categorical | Get dummies                  |
| •   Condition1: Proximity to main road or railroad                                       | Categorical | Not Used                     |
| •   Condition2: Proximity to main road or railroad (if a second is present)              | Categorical | Not Used                     |
| •   BldgType: Type of dwelling                                                           | Categorical | Not Used                     |
| •   HouseStyle: Style of dwelling                                                        | Categorical | Not Used                     |
| •   OverallQual: Overall material and finish quality                                     | Numerical   | None                         |
| •   OverallCond: Overall condition rating                                                |             | Redundant                    |
| •   YearBuilt: Original construction date                                                | Numerical   | None                         |
| •   YearRemodAdd: Remodel date (same as construction date if no remodeling or additions) | Numerical   | None                         |
| •   RoofStyle: Type of roof                                                              | Categorical | Not Used                     |
| •   RoofMatl: Roof material                                                              | Categorical | Not Used                     |
| •   Exterior1st: Exterior covering on house                                              | Categorical | Not Used                     |
| •   Exterior2nd: Exterior covering on house (if more than one material)                  | Categorical | Not Used                     |
| •   MasVnrType: Masonry veneer type                                                      | Categorical | Not Used                     |
| •   MasVnrArea: Masonry veneer area in square feet                                       | Numerical   | None                         |
| •   ExterQual: Exterior material quality                                                 | Categorical | Map to numerical             |
| •   ExterCond: Present condition of the material on the exterior                         | Categorical | Redundant                    |
| •   Foundation: Type of foundation                                                       | Categorical | Not Used                     |
| •   BsmtQual: Height of the basement                                                     | Categorical | Map to numerical             |
| •   BsmtCond: General condition of the basement                                          | Categorical | Redundant                    |
| •   BsmtExposure: Walkout or garden level basement walls                                 | Categorical | Map to numerical             |
| •   BsmtFinType1: Quality of basement finished area                                      | Categorical | Map to numerical             |
| •   BsmtFinSF1: Type 1 finished square feet                                              | Numerical   | None                         |
| •   BsmtFinType2: Quality of second finished area (if present)                           | Categorical | Map to numerical             |
| •   BsmtFinSF2: Type 2 finished square feet                                              | Numerical   | None                         |
| •   BsmtUnfSF: Unfinished square feet of basement area                                   | Numerical   | None                         |
| •   TotalBsmtSF: Total square feet of basement area                                      | Numerical   | None                         |
| •   Heating: Type of heating                                                             | Categorical | Not Used                     |
| •   HeatingQC: Heating quality and condition                                             | Categorical | Map to numerical             |
| •   CentralAir: Central air conditioning                                                 | Categorical | Map to numerical             |
| •   Electrical: Electrical system                                                        | Categorical | Not Used                     |
| •   1stFlrSF: First Floor square feet                                                    | Numerical   | None                         |
| •   2ndFlrSF: Second floor square feet                                                   | Numerical   | None                         |
| •   LowQualFinSF: Low quality finished square feet (all floors)                          | Numerical   | None                         |
| •   GrLivArea: Above grade (ground) living area square feet                              | Numerical   | None                         |
| •   BsmtFullBath: Basement full bathrooms                                                | Numerical   | None                         |
| •   BsmtHalfBath: Basement half bathrooms                                                | Numerical   | None                         |
| •   FullBath: Full bathrooms above grade                                                 | Numerical   | None                         |
| •   HalfBath: Half baths above grade                                                     | Numerical   | None                         |
| •   Bedroom: Number of bedrooms above basement level                                     | Numerical   | None                         |
| •   Kitchen: Number of kitchens                                                          | Numerical   | None                         |
| •   KitchenQual: Kitchen quality                                                         | Categorical | Map to numerical             |
| •   TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)                   | Numerical   | None                         |
| •   Functional: Home functionality rating                                                | Categorical | Map to numerical             |
| •   Fireplaces: Number of fireplaces                                                     | Numerical   | None                         |
| •   FireplaceQu: Fireplace quality                                                       | Categorical | Map to numerical             |
| •   GarageType: Garage location                                                          | Categorical | Not Used                     |
| •   GarageYrBlt: Year garage was built                                                   | Numerical   | None                         |
| •   GarageFinish: Interior finish of the garage                                          | Categorical | Map to numerical             |
| •   GarageCars: Size of garage in car capacity                                           | Numerical   | None                         |
| •   GarageArea: Size of garage in square feet                                            | Numerical   | None                         |
| •   GarageQual: Garage quality                                                           | Categorical | Map to numerical             |
| •   GarageCond: Garage condition                                                         | Categorical | Redundant                    |
| •   PavedDrive: Paved driveway                                                           | Categorical | Map to numerical             |
| •   WoodDeckSF: Wood deck area in square feet                                            | Numerical   | None                         |
| •   OpenPorchSF: Open porch area in square feet                                          | Numerical   | None                         |
| •   EnclosedPorch: Enclosed porch area in square feet                                    | Numerical   | None                         |
| •   3SsnPorch: Three season porch area in square feet                                    | Numerical   | None                         |
| •   ScreenPorch: Screen porch area in square feet                                        | Numerical   | None                         |
| •   PoolArea: Pool area in square feet                                                   | Numerical   | None                         |
| •   PoolQC: Pool quality                                                                 | Categorical | Map to numerical             |
| •   Fence: Fence quality                                                                 | Categorical | Not Used                     |
| •   MiscFeature: Miscellaneous feature not covered in other categories                   | Categorical | Not Used                     |
| •   MiscVal: $Value of miscellaneous feature                                             | Numerical   | None                         |
| •   MoSold: Month Sold                                                                   | Numerical   | None                         |
| •   YrSold: Year Sold                                                                    | Numerical   | None                         |
| •   SaleType: Type of sale                                                               | Categorical | Future mapping to numerical? |






## Conclusions and Recommendations

<font size='2'>
With $R^2$ scores of around 0.83, the linear regressions run on this particular dataset, with the null-value-filling and selective multicollinear column dropping done here has had a relatively robust result.  With the usual caveats that the underlying drivers could change in the future, we can nevertheless provide some concrete interpretations and recommendations regarding the Ames, IA housing market.  

Please refer to the notebook cells in the "Conclusions and Recommendations" Section at the end.  In that section's cells, the absolute values of the coefficients that are the 25 most impactful are listed.  To get the direction of the impact (positive or negative) look at the two cells below that.  The impacts can be as high as paying an additional `$`111,000 to live in Green Hills  versus the Somerset neighborhood (`$`116,000 - `$`15,000), for a house that is identical in all other respects.  

Outside of location, there are also specific types of quality improvements that are impactful on Sale Price, such as Basement Quality, Garage Quality, and Exterior Quality.  Making one-level improvements on the quality scale can net the owner `$`9,500 - `$`13,000 per quality category.  

In looking at the p-values, we can also see that Total above-grade square footage, along with Overall Quality are the highest correlated metrics to Sale Price.  The coefficients might be lower, but that is only on an unscaled basis.  All in all, the linear regressions have generated inferences that are intuitive and actionable.

</font>









