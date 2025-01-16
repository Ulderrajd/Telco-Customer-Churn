# Telco-Customer-Churn
# Project Brackground

<aside>
🧑‍💻 Full code: [Telco Customer Churn](https://colab.research.google.com/drive/16dNaAXFE03UIdsiP0ahy0YgcIF9Y9hMs#scrollTo=emx-OGaDyZFQ)

</aside>

This dataset provides information about a telecommunications company operating in California. The data covers a period of three months and includes details on 7043 customers who subscribed to home phone and internet services. Specifically, it reveals which customers discontinued their service (churned), maintained their subscription, or newly signed up for the company's services.

This project aims to gain valuable insights from the dataset. Through these insights, company can adjust their service to retain its customer. Also, this project conducts a ML projection to predict potential churn customer.



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


# Summary & Recommendations

## Summary


Overall, there two main reasons leading the customers to churn, which is the good performance of others, as well as the poor service Telco providing, and the bad customer-facing service.

Regarding the poor service of Telco, specifically Internet Service they providing, we can see Fiber Optic, known as the best option as its provide faster Internet than Cable and DSL, has the most customers, as well as a massive churn rate. This is also reflected on the Churn Reason, as Telco has worse devices, less data with lower download speeds.

Another issue is the offers made to the customers, as they are not appealing, reflecting on the churn reason ‘Competitor made better offers’ and the massive figures of customers not having received any offers. Additionally, the offer E even has more churned customers than un-churned ones.


## Recommendations

- Internet Service: Telco's fiber optic infrastructure development and growth must be given top priority. Given its better reliability over cable and DSL, a strong fiber optic network is necessary to satisfy customer demands for dependability and speed. It's also critical to address issues with data limits and device compatibility. Customer happiness will be greatly increased by upgrading hardware, providing quicker speeds, and giving more flexible data plans.
- Customer-Facing Service: It is critical to raise the standard of customer service. This entails giving employees thorough training so they can respond to consumer requests, address problems quickly, and offer tailored solutions. Their experience will be improved by cutting down on wait times and making sure that consumers are communicated with effectively. To preserve openness and foster consumer trust, proactive communication about maintenance plans, upgrades, and service interruptions is essential.
- Making Attractive deals: Telco needs to provide deals that are both attractive and offer customers genuine value. This can entail combining services, putting loyalty plans in place, and providing tailored discounts according to the requirements and usage habits of each particular client.
- Proactive Engagement: It's critical to proactively contact clients with offers that are customized to meet their unique requirements and preferences. This individualized approach will promote customer retention and show a sincere concern in customer happiness.
