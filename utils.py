import openai
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import *
from io import StringIO
import sys
import os
import re

custom_table_info = {
    "pnl_statement": """CREATE TABLE pnl_statement (
    id INTEGER PRIMARY KEY,
    account_type TEXT,
    account_category TEXT,
    account_sub_category TEXT,
    amount REAL,
    year INTEGER,
    month INTEGER
);

/*
For salary related queries look at following account_sub_category:  
Expenses	   62000 Personnel & Staffing	         62240 Executive  
Expenses	   62000 Personnel & Staffing	         62250 Marketing  
Expenses	   62000 Personnel & Staffing	         62291 Payroll Processing Fees  
Expenses	   62000 Personnel & Staffing	         62292 Payroll Taxes 


68 rows from pnl_statement table:
account_type	account_category	account_sub_category Description
Income	   41001 Wholesale Sales	      41010 Grocery
Income	   41002 Wholesale Sales	      41990 Discounts Given
Income	   42000 DTC Sales	   42000 DTC Sales
Income	   42001 DTC Sales	      42020 Shopify
Income	   42002 DTC Sales	      42980 Returns & Refunds
Income	   42003 DTC Sales	      42990 Discounts given
Cost of Goods Sold	   50001 Cost of Goods Sold	      51000 Freight-In
Cost of Goods Sold	   50002 Cost of Goods Sold	      52000 Freight-Out
Cost of Goods Sold	   50003 Cost of Goods Sold	      54000 Packaging & Labeling
Cost of Goods Sold	   50004 Cost of Goods Sold	      55000 Product Costs & Materials
Expenses	   61000 General & Administrative	      61030 Bank Charges
Expenses	   61000 General & Administrative	      61050 Continuing Education
Expenses	   61000 General & Administrative	      61100 Equipment (Under $1,000)
Expenses	   61000 General & Administrative	         61210 D&O/Liability Insurance
Expenses	   61000 General & Administrative	         61220 Health Insurance
Expenses	   61000 General & Administrative	         61230 Workers Compensation Insurance
Expenses	   61000 General & Administrative	      61300 Interest & Fees
Expenses	   61000 General & Administrative	      61400 Meals & Entertainment (100% Deductible)
Expenses	   61000 General & Administrative	      61450 Merchant Fees
Expenses	   61000 General & Administrative	      61500 Office Supplies
Expenses	   61000 General & Administrative	         61610 Accounting
Expenses	   61000 General & Administrative	         61620 Bookkeeping
Expenses	   61000 General & Administrative	         61630 Legal
Expenses	   61000 General & Administrative	      61700 Rent
Expenses	   61000 General & Administrative	      61800 Software & Online Services
Expenses	   61000 General & Administrative	      61850 Storage & Warehousing
Expenses	   61000 General & Administrative	         61851 Internal Freight
Expenses	   61000 General & Administrative	      61900 Taxes Paid
Expenses	   61000 General & Administrative	      61950 Travel
Expenses	   61000 General & Administrative	         61951 Airfare
Expenses	   61000 General & Administrative	         61952 Local Transportation
Expenses	   61000 General & Administrative	         61953 Lodging
Expenses	   61000 General & Administrative	         61954 Parking & Tolls
Expenses	   61000 General & Administrative	      61990 Utilities
Expenses	   61000 General & Administrative	   61075 Employee Recruitment
Expenses	   62000 Personnel & Staffing	      62100 Consultants
Expenses	   62000 Personnel & Staffing	         62105 Customer Service
Expenses	   62000 Personnel & Staffing	         62110 Design
Expenses	   62000 Personnel & Staffing	         62120 Engineering
Expenses	   62000 Personnel & Staffing	         62125 General
Expenses	   62000 Personnel & Staffing	         62130 Product Development
Expenses	   62000 Personnel & Staffing	      62200 Payroll
Expenses	   62000 Personnel & Staffing	         62240 Executive  Salary
Expenses	   62000 Personnel & Staffing	         62250 Marketing  Salary
Expenses	   62000 Personnel & Staffing	         62291 Payroll Processing Fees  Salary
Expenses	   62000 Personnel & Staffing	         62292 Payroll Taxes Salary
Expenses	   63000 Sales & Marketing	      63100 Broker Commissions
Expenses	   63000 Sales & Marketing	      63150 Consultants
Expenses	   63000 Sales & Marketing	      63200 Content Production
Expenses	   63000 Sales & Marketing	      63300 Distributor Fees
Expenses	   63000 Sales & Marketing	      63350 Market Research
Expenses	   63000 Sales & Marketing	      63400 Paid Media Agency Fees
Expenses	   63000 Sales & Marketing	      63450 Paid Media Influencer
Expenses	   63000 Sales & Marketing	      63500 Paid Media Spend
Expenses	   63000 Sales & Marketing	      63600 Promotional Items
Expenses	   63000 Sales & Marketing	      63700 Public Relations
Expenses	   63000 Sales & Marketing	      63800 Reviews & Product Certifications
Expenses	   63000 Sales & Marketing	      63900 Social Media and Influencers
Expenses	   63000 Sales & Marketing	      63950 Software
Expenses	   63000 Sales & Marketing	      63955 Website Related
Expenses	   63000 Sales & Marketing	      63960 Wholesale Chargebacks
Other Income	   Credit card rewards	   Credit card rewards
Other Income	   DADZ  Balance Transfers	   DADZ  Balance Transfers
Other Income	   Interest or Dividends Earned	   Interest or Dividends Earned
Other Income	   Other Income	   Other Income
Other Expenses	   Unrealized Gain or Loss	   Unrealized Gain or Loss
Other Expenses	   Inventory Write-Down	   Inventory Write-Down
*/"""
}

model_id = "gpt-4"

os.environ["OPENAI_API_KEY"] = ""
db = SQLDatabase.from_uri("sqlite:///PocketCFO.db", include_tables=['pnl_statement'],
                          custom_table_info=custom_table_info)

llm = ChatOpenAI(temperature=0, model=model_id)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True

)

agent_executor.return_intermediate_steps = True

def get_initial_message():
    messages = [
        {"role": "system", "content": "You are a helpful AI Tutor. Who anwers brief questions about AI."},
        {"role": "user", "content": "I want to learn AI"},
        {"role": "assistant", "content": "Thats awesome, what do you want to know aboout AI"}
    ]
    return messages


def get_chatgpt_response(messages, model="gpt-3.5-turbo"):
    agent_response = []

    response = agent_executor(messages)
    agent_response.append(f"<b>{response['output']}</b>")

    for step in response["intermediate_steps"]:
        agent_response.append(step[0][2])

    print(agent_response)

    return "\n".join(agent_response).replace("\n", "<br><br>")


def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages
