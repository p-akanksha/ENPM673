# Homework 1

## Ordinary Least Square 

The python script ls.py accepts the path to the csv file containing the data set (Default value = /home/akanksha/Documents/ENPM673/apatel44_hw1/data_1.csv).

Run the following command:
```
python3.5 ls.py --DataPath <path_to_csv_file>
```

The best fitting curve is displayed on the screen and saved in the Results folder.

## RANSAC

The python script RANSAC.py accepts the following parameters:

1. DataPath - path to the csv file containing the data set (Default value = /home/akanksha/Documents/ENPM673/apatel44_hw1/data_2.csv)
2. k - Number of iterations (Default = 100)
3. t - Threshold value that deternime that a data point is inlier or outlier (Default = 40)
4. d - Minimum number of inliers for a model to be selected as a good fit (Default = 175)

Run the following command:
```
python3.5 RANSAC.py --DataPath <path_to_csv_file> --k <number_of_iterations> --t <threshold_value> --d <minimum_number_of_inliers>
```
The best fitting curve is displayed on the screen and saved in the Results folder.

## Singular Value Decomposition

Run the following command:

```
python3.5 svd.py
```
The U, D and V-transpose matrices are printed as output. 

