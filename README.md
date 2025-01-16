# Telco-Customer-Churn
# Project Brackground

<aside>
🧑‍💻 Full code: [Telco Customer Churn](https://colab.research.google.com/drive/16dNaAXFE03UIdsiP0ahy0YgcIF9Y9hMs#scrollTo=emx-OGaDyZFQ)

</aside>

This dataset provides information about a telecommunications company operating in California. The data covers a period of three months and includes details on 7043 customers who subscribed to home phone and internet services. Specifically, it reveals which customers discontinued their service (churned), maintained their subscription, or newly signed up for the company's services.

This project aims to gain valuable insights from the dataset. Through these insights, company can adjust their service to retain its customer. Also, this project conducts a ML projection to predict potential churn customer.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/e42c510b-7b8a-47a3-bf54-e1a38f88a4fe/b2b5451e-a2de-4cb1-9dce-ff184ec5adec.png)

# Data Structure

The dataset consists of 50 features and 7043 records indicating 7043 customers.

- Notable features:
    - Customer ID: Unique ID assigned to each customer
    - Customer Demographic
        - Gender: ‘Male’ or ‘Female’
        - Age: Customer’s age
        - Number of Dependents: Depedents can be seen as people living on the customer
    - Number of Referrals: Number of referrals made by the customer
    - Tenure in Months: How long the person has been a customer of Telco
    - Offer: Specific offer or promotion customer have received
    - Customer plan’s specifications: Detail information of customer’s subscription plan
        - Internet Service: Whether the customer has internet service (’Yes’ or ‘No’)
        - Internet Type: The type of internet service (DSL, Fiber optic, No).
        - AVG Monthly GB Download: The average number of GBs downloaded per month.
        - Online Security: Whether the customer has signed up for additional  online security service (’Yes’ or ‘No’)
        - Online Backup: Whether customer subscribe for online backup (’Yes’ or ‘No’)
        - Device Protection Plan: Whether customer subscribe for device protection (’Yes’ or ‘No’)
        - Premium Tech Support: Whether customer subscribe for premium tech support (’Yes’ or ‘No’)
        - Streaming TV: Whether the customer has signed up for streaming TV (’Yes’ or ‘No’)
        - Streaming Movies: Whether the customer has signed up for streaming movie (’Yes’ or ‘No’)
        - Streaming Music: Whether the customer has signed up for streaming music (’Yes’ or ‘No’)
        - Unlimited Data: Whether the customer has an unlimited data plan (’Yes’ or ‘No’)
    - Contract: Contract comprises ‘Month-to-month’, ‘One year’,  ‘Two years’ types
    - Payment method:
        - Paperless Billing: Whether customer use electronic or paper billing (’Yes’ or ‘No’)
        - Payment Method: Method of payment (’Bank Withdrawal’, ‘Credit Card’, ‘Mailed Check’)
    - Charges & Revenue gained from customers:
        - Monthly Charge
        - Total Charges
        - Total Refunds
        - Total Extra Data Charges
        - Total Long Distance Charges
        - Total Revenue
    - Satisfaction Score: A score that reflects the customer's satisfaction.
    - Churn Situation:
        - Customer Status: Current status of the customer (’Stayed’, ‘Churned’, or ‘Joined’)
        - Churn Label: Whether the customer has churned (’Yes’ or ‘No’)
        - Churn Score: A score predicting the likelihood of churn.
        - Churn Category: The reason for churn, categorized.
        - Churn Reason: Specific reason for churn
    - CLTV: Customer Lifetime Value
    

# Workflow

## Data Loading and Cleaning

**Import and read the dataset from Kaggle**

```
!kaggle datasets download rhonarosecortez/telco-customer-churn
!unzip telco-customer-churn.zip
df = pd.read_csv('TelcoCustomerChurn.csv')
df.head()
```

**Checking data type for the columns, as well as duplicated values**

```
print(df.info())
print('----------------------------------------------------------')
print(df.isnull().sum())
print('----------------------------------------------------------')
print(df.duplicated().sum())
```

All data types are assigned properly. There is no duplicated record. However, there is number of null values in specific columns.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/b8dcbfcf-07c1-466e-959d-6c74d8f592ad/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/37b0c919-3025-4f0f-8629-d7da4e409a47/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/9b94efec-a0ab-4489-ac08-c6250055386e/image.png)

**Handling Null Values**

After checking, it seems that the null values in 2 columns 'InternetType' and 'ChurnCategory' (as well as 'ChurnReason') are aligned with 'No' rows in 2 columns 'InternetService' and 'ChurnLabel', respectively. 

Check if whether there is null values in Internet Type for 'Yes' value InternetService; and whether there is null values in ChurnCategory, ChurnReason for 'Yes' value ChurnLabel:

```
#Check for null values in Internet Type for 'Yes' in Internet Service
print('null values for Yes value Internet Service: ',
      df[df['InternetService'] == 'Yes']['InternetType'].isnull().sum())
#Check for null values in Churn Category, Churn Reason for 'Yes' value in Churn Label
print('null values for Yes value Churn Label:\n',
      df[df['ChurnLabel'] == 'Yes'][['ChurnCategory', 'ChurnReason']].isnull().sum())

null values for Yes value Internet Service:  0
null values for Yes value Churn Label:
 ChurnCategory    0
ChurnReason      0
```

**Replace Null values**

```
df.fillna('No', inplace = True)
print(f'Null values: {df.isnull().sum().sum()}')

Null values: 0
```

**Dropping unecessary columns**

As there's already an 'Age' column, I drop 'Under30'

There's already a 'NumberofDependents' column, so I drop 'Dependents'

I drop 'Country' and 'State', 'Quarter', and 'ZipCode' as they are not necessary either for EDA or for ML.

## EDA

**Checking columns related to revenue:**

We can see that:

- Total Charges ~ Tenure * Monthly Charge
- Total Long Distance Charges = Tenure * AVG Long Distance Charge
- TotalRevenue = TotalCharges - TotalRefunds + TotalExtraDataCharges + TotalLongDistanceCharges

**Append features into different lists (numeric, object and coordinates)**

```
num = []
ob = []
location = ['Latitude', 'Longitude']
for col in df.columns:
  if col in location:
    continue
  elif df[col].dtype == 'object':
    ob.append(col)
  else:
    num.append(col)
ob.remove('CustomerID')
```

**Illustrating box plots for numeric columns by Churn Label**

```
def plot_boxplot(df, x,  color = '#5A9', hue = None):
  num_columns = len(df.columns)
  num_rows = (num_columns + 1) // 2

  fig, axes = plt.subplots(num_rows, 2, figsize = (16, 7 * num_rows))
  axes = axes.flatten()
  for i, column in enumerate(df.drop(columns = x).columns):
    sns.boxplot(data = df, x = x, y = column, hue = hue, ax = axes[i], color = color)
    axes[i].set_xlabel(x)
    axes[i].set_ylabel(column)
    axes[i].set_title(f'Boxplot of {column} by {x}')

  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout()
  plt.show()
plot_boxplot(df[num + ['ChurnLabel']], x = 'ChurnLabel')
```

We can gain some infomation based on these box plots:

- Number of Referrals: Staying customers tend to have more referrals than churned customers.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/e66b52ce-ff8c-4a29-b1c9-60faa18c44d4/image.png)

- TenureInMonths: churned customers tend to have shorter time using service from the company. While for the stayed customer, the pattern is opposite.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/ae630261-30d6-4fb6-a350-3b1a0dc7b37f/image.png)

- MonthlyCharge: Churned customers have the interquartile and the median value of monthly charge higher than those of staying customers.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/2381c56f-e1b5-46e9-b1c2-10e273d45f1a/image.png)

- TotalCharges and TotalRevenue:

The two features have the same pattern due to Total Revenue is determined by Total Charges according to the formula:

TotalRevenue = TotalCharges - TotalRefunds + TotalExtraDataCharges + TotalLongDistanceCharges

These two features have the opposite pattern from MonthlyCharge, company gains more revenue from the staying customers. This can be explained by that staying customers stick with the company longer, so that the records for TenureInMonths are longer, which leads to higher TotalCharges and TotalRevenue. 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/34a4600a-a1fb-4f65-97d1-9e1c385d6b2b/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/84ae6f1e-234a-4d6b-8fa5-1b91af853498/image.png)

- ChurnScore (score that predict the likelihood to churn): Churned customers have higher ChurnScore.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/39d4c4c4-e4fa-4f5d-b5c6-a5c44aae7b7c/image.png)

- SatisfactionScore: churned customer has lower SatisfactionScore.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/77497ff6-dd23-40ac-b2cf-0c76cf3cfbcd/image.png)

- CLTV (Customer Time Value): interquartile of stayed customers approximately distributed in the span of 3600 to 5300, while for the churned customers, interquatile is in the span of approximately 3100 to about 5100. The span of CLTV figures of churned and un-churned are not much of a gap, which emphasis the importance reducing possible churned customer.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/b3ced3ad-9441-4fb7-b49c-240528f0b135/image.png)

**Illustrating count plot for categorical features by ChurnLabel**

```
def plot_countplot(df, palette = 'dark:#5A9_r', hue = None):
  num_columns = len(df.columns)
  num_rows = (num_columns + 1) // 3

  fig, axes = plt.subplots(num_rows, 3, figsize = (16, 5 * num_rows))
  axes = axes.flatten()

  for i, column in enumerate(df.columns):
    ax = sns.countplot(data = df, x = column, hue = hue, ax = axes[i], palette = palette)
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')
    axes[i].set_title(f'Countplot of {column} by {hue}')

  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout()
  plt.show()
plot_countplot(df[ob].drop(columns = ['ChurnReason','City']), hue = 'ChurnLabel')

```

Notable findings:

- Contract: most of the churned customers are likely to have month-to-month contract

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/60628834-9c6e-4118-adbb-eaa8cad19e90/84414979-c363-4510-827d-45da26faedeb.png)

- Internet Service: The proportion of churned customer having internet service is suprisingly much higher than those not having internet service. This raise a consideration of Telco’s Internet Service.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/985e8dac-2ea4-4524-9c42-96e5234495af/25c6460d-0747-4166-9d12-3ba767bdf23a.png)

- Among the Internet Service Type (InternetType): Fiber Optic has a enourmous proportion of churned customer, which explain the high churn rate of Internet Service features.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/896778f1-4838-4ed4-834a-e27507267bba/image.png)

- Offer:
    - Most customers don't take any offer
    - The number of churned customers who receive Offer E is larger than that of stayed customers. Even for customers who don't receive any offer, the number of staying customers is larger than the number of churned ones.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/e62d1534-3d95-46d4-a4b9-961caeb9fe8c/image.png)
    
- Count plot for Churn Reason

```
sns.countplot(data = df[df['ChurnReason'] != 'No'], y = 'ChurnReason', color = '#5A9',
              order = df[df['ChurnReason'] != 'No']['ChurnReason'].value_counts().index)
plt.ylabel('Churn Reason')
plt.xlabel('Count')
plt.title('Countplot of Churn Reason')
plt.show()
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/9c6b9127-d69e-4c81-80d2-cb6128517e48/image.png)

According to the plot, most counts of the reason churning are relating to the competitor better performance and poor attitude of the support and service providing teams.

This emphasize the competitiveness of the competitor, as well as the poor performance of the customer-facing staffs.

## Feature Engineering and Selecting

**Correlation Matrix**

Previously, we have already known:

- Total Charges ~ Tenure * Monthly Charge
- Total Long Distance Charges = Tenure * AVG Long Distance Charge
- TotalRevenue = TotalCharges - TotalRefunds + TotalExtraDataCharges + TotalLongDistanceChargesIt is obviously that those columns Total Charges, Total Long Distance Charges, Total Revenue are highly correlated with some other columns in the revenue gained from customers. Therefore, I will omit these features from the correlation matrix and also from the machine learning model to avoid multicollinearity.

```
num.remove('TotalCharges')
num.remove('TotalLongDistanceCharges')
num.remove('TotalRevenue')
correlation = df[num].corr()
plt.figure(figsize = (16, 8))
sns.heatmap(correlation, annot = True, cmap = 'coolwarm')
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/cc0bc730-8733-4f4b-949d-10570bad1dbb/image.png)

Notable correlation:

- Age is negatively correlated with Average Monthly Download
- Number of Referrals is positively correlated with Number of Dependents and Tenure
- Tenure is positive correlated with CLTV
- Average Monthly GB Download is positive correlated with Monthly Charge, obviously
- Monthly Charge is positively correlated with Tenure -> it can indicate that longer lifetime customer tends to pay more
- Satisfaction Score is positive correlated with Number of Referrals, Tenure and negatively correlated with Monthly Charge
- Churn Score is negatively correlated with Satisfaction Score

**Applying Encoder and Standard Scaler:**

- Apply One Hot Encoder for discrete data features having more than 2 values
- Apply Label Encoder for feature with continuous data ('Contract')
- Remove City, ChurnReason, CustomerStatus, and Churn Category

```
ob.remove('City')
ob.remove('ChurnReason')
ob.remove('CustomerStatus')
ob.remove('ChurnCategory')
ob_dummies = []
ob_ohe = []
ob_le = []
for col in ob:
  if col == 'Contract':
    ob_le.append(col)
  else:
    ob_ohe.append(col)
```

One Hot Encode variables:

```
ohe = OneHotEncoder(sparse_output = False, drop = 'if_binary')
encoded_cols = ohe.fit_transform(df[ob_ohe])
ohe_df = pd.DataFrame(encoded_cols, columns = ohe.get_feature_names_out(ob_ohe))
```

Label Encode for ‘Contract’:

```
le = LabelEncoder()
oble = le.fit_transform(df[ob_le])
```

Standard Scaler:

```
ss = StandardScaler()
scaled_cols = ss.fit_transform(df[num])
scaled_df = pd.DataFrame(scaled_cols, columns = num)
```

Create a new dataframe with encoded and scaled features:

```
new_df = pd.concat([scaled_df, ohe_df, pd.Series(oble, name = 'Contract')], axis = 1)
```

Applying Chi-squared test for Categorical features:

```
new_df = new_df.rename(columns = {'ChurnLabel_Yes' : 'ChurnLabel'})
X = new_df.drop(columns = num + ['ChurnLabel'])
y = new_df['ChurnLabel']
chi_scores = chi2(X, y)
p_values = pd.Series(chi_scores[1], index = X.columns)
p_values.sort_values(ascending = False, inplace = True)
p_values.plot.bar()
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/2633b966-1f1c-4c25-85d1-fa26eb1c4ee9/image.png)

Omit features with p-values > 0.05

```jsx
new_df = new_df.drop(columns = p_values[p_values > 0.05].index)
```

## Machine Learning: Classification

I use ChurnLabel as the dependent variable.

**Balancing the dataset**

```
sns.countplot(data = new_df, x = 'ChurnLabel', color = '#5A9')
plt.ylabel('Count')
plt.xlabel('Churn Label')
plt.title('Countplot of Churn Label')
plt.show()
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/d46cf81c-8355-409b-a7c8-08a8fdd47446/image.png)

```
x = new_df.drop(columns = 'ChurnLabel')
y = new_df['ChurnLabel']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
```

Using SMOTE to balance 0 and 1 values of the dependent variable

```
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
counter_before = Counter(y)
counter_after = Counter(y_train_resampled)
df_after = pd.DataFrame(counter_after.items(), columns = ['ChurnLabel', 'Count'])
sns.countplot(data = df_after, x = 'ChurnLabel', color = '#5A9')
plt.show()

```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/ab9f9fc6-4ea3-494a-8bb7-012515e42b19/image.png)

**Logistic Regression:**

```
lr = LogisticRegression()
lr.fit(X_train_resampled, y_train_resampled)
lr_pred = lr.predict(X_test)
ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test)
plt.show()
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
print('Accuracy score:', accuracy_score(y_test, lr_pred))

```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/200f82ca-02ee-400f-95b2-7eb32d7f39e6/image.png)

Decision Tree Classification:

```jsx
dt = DecisionTreeClassifier()
dt.fit(X_train_resampled, y_train_resampled)
dt_pred = dt.predict(X_test)
ConfusionMatrixDisplay.from_estimator(dt, X_test, y_test)
plt.show()
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test,dt_pred))
print('Accuracy score:', accuracy_score(y_test, dt_pred))
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/8466a551-575d-43ca-9e2c-9ed85dda1a3d/image.png)

# Summary & Recommendations

## Summary

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/9c6b9127-d69e-4c81-80d2-cb6128517e48/image.png)

Overall, there two main reasons leading the customers to churn, which is the good performance of others, as well as the poor service Telco providing, and the bad customer-facing service.

Regarding the poor service of Telco, specifically Internet Service they providing, we can see Fiber Optic, known as the best option as its provide faster Internet than Cable and DSL, has the most customers, as well as a massive churn rate. This is also reflected on the Churn Reason, as Telco has worse devices, less data with lower download speeds.

Another issue is the offers made to the customers, as they are not appealing, reflecting on the churn reason ‘Competitor made better offers’ and the massive figures of customers not having received any offers. Additionally, the offer E even has more churned customers than un-churned ones.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/06363598-db09-4d9e-bea4-8cc96d7cbef7/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/38bd8733-d67c-43c8-8252-e1f565b194c7/e62d1534-3d95-46d4-a4b9-961caeb9fe8c/image.png)

## Recommendations

- Internet Service: Telco's fiber optic infrastructure development and growth must be given top priority. Given its better reliability over cable and DSL, a strong fiber optic network is necessary to satisfy customer demands for dependability and speed. It's also critical to address issues with data limits and device compatibility. Customer happiness will be greatly increased by upgrading hardware, providing quicker speeds, and giving more flexible data plans.
- Customer-Facing Service: It is critical to raise the standard of customer service. This entails giving employees thorough training so they can respond to consumer requests, address problems quickly, and offer tailored solutions. Their experience will be improved by cutting down on wait times and making sure that consumers are communicated with effectively. To preserve openness and foster consumer trust, proactive communication about maintenance plans, upgrades, and service interruptions is essential.
- Making Attractive deals: Telco needs to provide deals that are both attractive and offer customers genuine value. This can entail combining services, putting loyalty plans in place, and providing tailored discounts according to the requirements and usage habits of each particular client.
- Proactive Engagement: It's critical to proactively contact clients with offers that are customized to meet their unique requirements and preferences. This individualized approach will promote customer retention and show a sincere concern in customer happiness.
