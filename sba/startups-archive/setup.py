import ipywidgets as widgets
from IPython.display import display

import pandas as pd
import numpy as np

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import seaborn as sns

# load acquisitions data
acquisitions = pd.read_csv("acquisitions.csv")

# drop unnecessary columns
acquisitions.drop(columns=[
    'id', 'acquisition_id', 'created_at', 'updated_at', 
    'source_url', 'source_description'
], inplace=True)

# convert columns
acquisitions['term_code'] = acquisitions['term_code'].astype('category')
acquisitions['price_currency_code'] = acquisitions['price_currency_code'].astype('category')
acquisitions['acquired_at'] = pd.to_datetime(acquisitions['acquired_at'])

# remove duplicates
acquisitions = acquisitions.drop_duplicates()

# load funding_rounds data
funding_rounds = pd.read_csv("funding_rounds.csv")

# drop unnecessary columns
funding_rounds.drop(columns=[
    'id', 'created_at', 'updated_at', 'created_by',
    'raised_amount', 'raised_currency_code', 'pre_money_valuation', 'post_money_valuation_usd',
    'pre_money_currency_code', 'post_money_valuation', 'pre_money_valuation_usd',
    'post_money_currency_code', 'source_url'
], inplace=True)

# convert data types
funding_rounds['funded_at'] = pd.to_datetime(funding_rounds['funded_at'])
categorical_cols = ['funding_round_type', 'funding_round_code', 'is_first_round', 'is_last_round']
for col in categorical_cols:
    funding_rounds[col] = funding_rounds[col].astype('category')

# fix funding round type inconsistencies
condition = (funding_rounds['funding_round_code'] == "angel") & (funding_rounds['funding_round_type'] == "series-a")
funding_rounds.loc[condition, 'funding_round_type'] = "angel"

# load investments data
investments = pd.read_csv("investments.csv")
investments.drop(columns=['id', 'created_at', 'updated_at'], inplace=True)
investments = investments.drop_duplicates()

# load IPOs data
ipos = pd.read_csv("ipos.csv")
ipos.drop(columns=[
    'id', 'created_at', 'updated_at', 'stock_symbol',
    'source_url', 'source_description'
], inplace=True)
ipos['valuation_currency_code'] = ipos['valuation_currency_code'].astype('category')
ipos['raised_currency_code'] = ipos['raised_currency_code'].astype('category')
ipos['public_at'] = pd.to_datetime(ipos['public_at'])
ipos = ipos.drop_duplicates()

# load milestones data
milestones = pd.read_csv("milestones.csv")
milestones.drop(columns=[
    'id', 'created_at', 'updated_at', 'source_url', 'milestone_code'
], inplace=True)
milestones['milestone_at'] = pd.to_datetime(milestones['milestone_at'])
milestones = milestones.drop_duplicates()

# load objects data
objects = pd.read_csv("objects.csv")
objects.drop(columns=[
    'normalized_name', 'permalink', 'created_at', 'updated_at', 'first_investment_at',
    'last_investment_at', 'created_by', 'domain', 'twitter_username', 'logo_url', 'overview'
], inplace=True)

# convert columns
objects['founded_at'] = pd.to_datetime(objects['founded_at'])
objects['logo_width'] = pd.to_numeric(objects['logo_width'], errors='coerce')
objects['logo_height'] = pd.to_numeric(objects['logo_height'], errors='coerce')
objects['country_code'] = objects['country_code'].astype('category')
objects['state_code'] = objects['state_code'].astype('category')
objects['investment_rounds'] = objects['investment_rounds'].astype('Int64')
objects['invested_companies'] = objects['invested_companies'].astype('Int64')
objects['first_funding_at'] = pd.to_datetime(objects['first_funding_at'])
objects['last_funding_at'] = pd.to_datetime(objects['last_funding_at'])
objects['funding_rounds'] = objects['funding_rounds'].astype('Int64')
objects['funding_total_usd'] = pd.to_numeric(objects['funding_total_usd'], errors='coerce')
objects['first_milestone_at'] = pd.to_datetime(objects['first_milestone_at'])
objects['milestones'] = objects['milestones'].astype('Int64')
objects['relationships'] = pd.to_numeric(objects['relationships'], errors='coerce')

# load offices data
offices = pd.read_csv("offices.csv")
offices.drop(columns=['id', 'zip_code', 'created_at', 'updated_at'], inplace=True)
offices['state_code'] = offices['state_code'].astype('category')
offices = offices.drop_duplicates()

# load people data
people = pd.read_csv("people.csv")
people.drop(columns=['id'], inplace=True)
people = people.drop_duplicates()

# filter objects to create STARTUPS
STARTUPS = objects[(objects['entity_type'] == "Company") & 
                   (objects['status'] != "") & 
                   (objects['country_code'] != "CSS") & 
                   (objects['country_code'] != "FST")].copy()

# drop the 'entity_id' column and remove duplicates
STARTUPS.drop(columns=['entity_id'], inplace=True, errors='ignore')
STARTUPS = STARTUPS.drop_duplicates()

finale = pd.merge(STARTUPS, ipos, left_on="id", right_on="object_id", how="left")

# drop unnecessary columns
finale.drop(columns=[
    'homepage_url', 'parent_id', 'entity_type', 'short_description', 'description',
    'tag_list', 'valuation_amount', 'valuation_currency_code', 'raised_amount',
    'public_at', 'raised_currency_code', 'ipo_id'
], inplace=True, errors='ignore')

# merge/filter funding rounds
rounds = pd.merge(finale[['id']], funding_rounds, left_on='id', right_on='object_id', how='left')
rounds = rounds[['id', 'funded_at', 'funding_round_id', 'funding_round_type', 'raised_amount_usd']]
rounds = rounds.dropna(subset=['funding_round_type'])

# group by funding type and aggregate sums
rounds_summary = rounds.groupby('id').apply(lambda df: pd.Series({
    'angel': df.loc[df['funding_round_type'] == 'angel', 'raised_amount_usd'].sum(),
    'crowdfunding': df.loc[df['funding_round_type'] == 'crowdfunding', 'raised_amount_usd'].sum(),
    'other': df.loc[df['funding_round_type'] == 'other', 'raised_amount_usd'].sum(),
    'post_ipo': df.loc[df['funding_round_type'] == 'post-ipo', 'raised_amount_usd'].sum(),
    'private_equity': df.loc[df['funding_round_type'] == 'private_equity', 'raised_amount_usd'].sum(),
    'series_a': df.loc[df['funding_round_type'] == 'series-a', 'raised_amount_usd'].sum(),
    'series_b': df.loc[df['funding_round_type'] == 'series-b', 'raised_amount_usd'].sum(),
    'series_c': df.loc[df['funding_round_type'] == 'series-c+', 'raised_amount_usd'].sum(),
    'venture': df.loc[df['funding_round_type'] == 'venture', 'raised_amount_usd'].sum()
})).reset_index()

# merge funding summaries back
finale = pd.merge(finale, rounds_summary, on='id', how='left')

# number of acquisitions made
acq_count = acquisitions.groupby('acquiring_object_id').size().reset_index(name='num_acquisizioni_effettuate')
finale = pd.merge(finale, acq_count, left_on='id', right_on='acquiring_object_id', how='left')
finale['num_acquisizioni_effettuate'] = finale['num_acquisizioni_effettuate'].fillna(0)

# whether the company has been acquired
acquired_flags = acquisitions[['acquired_object_id']].copy()
acquired_flags['have_been_acquired'] = 1
acquired_flags.drop_duplicates(inplace=True)
finale = pd.merge(finale, acquired_flags, left_on='id', right_on='acquired_object_id', how='left')
finale['have_been_acquired'] = finale['have_been_acquired'].fillna(0)

# create FINANCIAL_ORG from objects
FINANCIAL_ORG = objects[objects['entity_type'] == "FinancialOrg"].copy()

FINANCIAL_ORG.drop(columns=[
    'closed_at', 'entity_id', 'parent_id', 'category_code', 
    'status', 'funding_rounds', 'funding_total_usd', 
    'first_funding_at', 'last_funding_at', 'milestones', 
    'last_milestone_at', 'first_milestone_at'
], inplace=True, errors='ignore')

# merge investments with financial organizations
t = pd.merge(investments, FINANCIAL_ORG, left_on='investor_object_id', right_on='id', how='inner')
t = t[['funded_object_id', 'investor_object_id']].groupby('funded_object_id').size().reset_index(name='n')
t['fin_org_financed'] = 1

finale = pd.merge(finale, t[['funded_object_id', 'fin_org_financed']], left_on='id', right_on='funded_object_id', how='left')

# drop unused or redundant columns
finale.drop(columns=[
    'first_milestone_at', 'last_milestone_at', 'last_funding_at',
    'first_funding_at', 'name', 'city', 'region', 'have_been_acquired',
    'closed_at', 'state_code'
], inplace=True, errors='ignore')

# drop duplicates
finale = finale.drop_duplicates()

# convert categorical columns
for col in ['category_code', 'status', 'country_code', 'fin_org_financed', 'person_financed', 'startup_financed']:
    if col in finale.columns:
        finale[col] = finale[col].astype('category')

finale.to_csv('startups.csv')