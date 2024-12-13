
# **D100-D400 Combined Project**

## **Project Overview**
This project focuses on predicting the price of used cars based on various features such as mileage, engine type, seating capacity, and more. The project involves:
1. **Data loading**
2. **Exploratory data analysis (EDA)**
3. **Data cleaning**
4. **Feature engineering**
5. **Model building and evaluation** using GLMs and LGBMs

---

## **Project Structure**
## **Dataset**
The raw dataset is located at: data/raw_data.csv

- **Number of Records**: 2,000
- **Number of Features**: 17
- **Feature Types**:
  - **Numerical**: `Price`, `Year`, `Kilometer`, `Length`, `Width`, `Height`, etc.
  - **Categorical**: `Fuel Type`, `Transmission`, `Seller Type`, etc.

---

## **How to Use**

### **1. Install Dependencies**
Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate d100-d400-project
