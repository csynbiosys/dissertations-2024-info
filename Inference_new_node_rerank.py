#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging

import torch
from layers import Model
from torch_geometric.sampler import BaseSampler
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DynamicBatchSampler
from torch_geometric.utils import k_hop_subgraph
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import expit
from torch_geometric.nn.pool import knn_graph
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import pandas as pd
import pickle


import random

df_inv_avg_prob = pd.read_csv('investor_avg_likelihood.csv')
df_sig_categories = pd.read_csv('significant_categories.csv')
df_inv_avg_general_prob = pd.read_csv('investor_all_category_avg_likelihood.csv')
def chunks(lst, n):
    _ = []    

    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        _.append(lst[i:i + n])
        
    return _

def split_dataset(x0, x1, edge_index, prop = 0.01):
    d = HeteroData()
    d['inv'].x = torch.FloatTensor(x0)
    d['org'].x = torch.FloatTensor(x1)
    
    d['inv', 'ii', 'org'].edge_index = torch.LongTensor(edge_index)
    
    return d

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None.npz")
    parser.add_argument("--pca", default = "pca_model.pkl")
    parser.add_argument("--lr", default = "0.0001") #4.24330774357976e-05
    parser.add_argument("--weight_decay", default = "0.006331364160485268") #0.03137191337585493
    parser.add_argument("--k", default = "16") #15
    
    parser.add_argument("--model_path", default = "model_best_check.pth")

    parser.add_argument("--odir", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args






def process_startup_features(provided_startup_features, args):
    # Assuming the provided_startup_features is already a numpy array
    # 1. PCA transformation
    pca = pickle.load(open(args.pca, 'rb'))
    # X_transformed = pca.transform(provided_startup_features.reshape(1, -1))
    
    
    
    # # 2. Normalization
    # mean_pca = np.load('mean_pca_x1.npy')
    # std_pca = np.load('std_pca_x1.npy')
    # X_normalized = (X_transformed - mean_pca) / std_pca
    X = provided_startup_features
    
    variance_mask = np.load('variance_mask_x1.npy')  # Load the actual variance mask (ii) here
    Xm = np.load('Xm_x1.npy')  # Load the actual mean (Xm) here
    
    X = X[:, variance_mask]
    
    # 4. Zero-Centering (Mean Subtraction)
    X = X - Xm
    
    x0 = np.load(args.ifile)['x0']
    
    pca = pickle.load(open(args.pca, 'rb'))
    
    X_transformed = pca.transform(X)
    
    # 6. Normalization
    # Assuming you have saved and loaded mean_pca and std_pca from the training script
    # For now, let's create dummy variables for these; replace these with the actual loaded values
    mean_pca = np.load('mean_pca_x1.npy')  # Load the actual mean of PCA-transformed training data here
    std_pca = np.load('std_pca_x1.npy')  # Load the actual std of PCA-transformed training data here
    
    X_normalized = (X_transformed - mean_pca) / std_pca
    
    return torch.tensor(X_normalized, dtype=torch.float32)


def recommend(provided_startup_features, dic):
    args = parse_args()
    device = torch.device('cpu')
    PATH = args.model_path
    batch_size = 16

    # Load model
    model = Model().to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # Load data
    x0 = np.load(args.ifile)['x0']
    x1 = np.load(args.ifile)['x1']
    edge_index = np.load('data.npz')['edge_index'][::-1, :].copy()

    data = split_dataset(x0, x1, edge_index)
    data = T.ToUndirected()(data)
    edge_index = data['inv', 'ii', 'org'].edge_index
    x_new = process_startup_features(provided_startup_features, args)

    # Prediction setup
    ii = list(range(x0.shape[0]))
    #random.shuffle(ii)
    inv_ii = chunks(list(ii), batch_size)
    Y_pred = []
    for ix, ii in enumerate(inv_ii):
        if ix % 200 == 0:
            logging.info('on batch {}...'.format(ix))
        
        # get the graph to test against
        x_inv = x0[ii]
    
    
    

        _, edge_index_, _, _ = k_hop_subgraph(torch.LongTensor(np.array(ii)), 1, edge_index, directed=False, flow='target_to_source')
        
    
        org_ii_ = list(set(list(edge_index_[1].numpy())))
    
        d_sub = dict()
        d_sub['inv'] = torch.LongTensor(np.array(ii))
        d_sub['org'] = torch.LongTensor(np.array(org_ii_))
    
        batch = data.subgraph(d_sub)
    
        e = batch['inv', 'ii', 'org'].edge_index.clone()
        n_known_edges = e.shape[1]
        e[1] += batch['inv'].x.shape[0]
        
        x = torch.cat([batch['inv'].x, batch['org'].x, x_new], 0).to(device)
    
        knn_indices = knn_graph(torch.cat([batch['org'].x, x_new], 0), int(args.k))
        knn_indices += batch['inv'].x.shape[0]
    
        e = torch.cat([e, knn_indices], 1).to(device)
        
        edge_attr = torch.zeros((e.shape[1], 2))
        edge_attr[-knn_indices.shape[1]:, 1] = 1.
        edge_attr[:n_known_edges, 0] = 1.
        edge_attr = edge_attr.to(device)
    
        with torch.no_grad():
            y_pred = model(x, e, edge_attr, batch['inv'].x.shape[0], x_new.shape[0])
            # print(y_pred)
            # _.extend()
            
        Y_pred.append(y_pred.detach().cpu().numpy().T)
        
    
    
    # print(Y_pred)

    # Extracting the investor recommendations
    with open("investor_name_dict.json", "r") as file:
        investor_name_dict = json.load(file)

    # Get predictions for the startup and filter out investors with predictions greater than 0.5
    # for y in Y_pred:
    #     print(y.shape)
    # Before concatenating, remove the last array
    
    
    
    # Assuming Y_pred is a list of arrays and Y_pred_array is the concatenated 2D array

    Y_pred = Y_pred[:-1]
    Y_pred_array = np.concatenate(Y_pred, axis=0)
    Y_pred_probs = expit(Y_pred_array)
    
    investor_indices_with_high_predictions = np.where(expit(Y_pred_array)>0.7)[0] #add expit
    
    # Sort these investors based on the prediction values in descending order
    sorted_investor_indices = investor_indices_with_high_predictions[np.argsort(-Y_pred_array.flatten()[investor_indices_with_high_predictions])]
    prediction_score = -Y_pred_array.flatten()[investor_indices_with_high_predictions]
    prediction_score = prediction_score.tolist()
    prediction_score = sorted(prediction_score, reverse=True)
    # Take the top 10 investors
    top_100_investors = sorted_investor_indices[:100]
    top_10_investors = sorted_investor_indices[:10]
    top_100_score = prediction_score[:100]

    # Map indices to investor names and print
    recommended_investors_100 = [investor_name_dict[str(index)] for index in top_100_investors]
    recommended_investors_10 = [investor_name_dict[str(index)] for index in top_10_investors]
    
    
    categories = [category.strip() for category in dic['category_groups_list'][0].split(',')]
    
    # ALL CATEGORY RERANK
    updated_investor_score_dict = {}
    investor_score_dict = dict(zip(recommended_investors_100, top_100_score))

    for category in categories:
        assert category in df_inv_avg_general_prob.columns 
        for name, score in investor_score_dict.items():
            likelihood = df_inv_avg_general_prob[df_inv_avg_general_prob['investor_name'] == name][category]
            if likelihood.empty:
                updated_investor_score_dict[name] = score * 0.001
            else:
                updated_investor_score_dict[name] = score * likelihood.values[0]
    sorted_dict = dict(sorted(updated_investor_score_dict.items(), key=lambda item: item[1], reverse=True))
    rerank_investors_100 = list(sorted_dict.keys())[:100]
    rerank_investors_10 = list(sorted_dict.keys())[:10]
    
    # SIGNIFICANT CATEGORY BOOST RERANK - ONLY TRIGGER IF STARTUP IS POPULAR CATEGORY GROUP
    if any(category in df_sig_categories.iloc[-1].values for category in categories):
        assert len(recommended_investors_100) == len(top_100_score)
        for name, score in investor_score_dict.items():
            avg_likelihood = df_inv_avg_prob[df_inv_avg_prob['investor_name'] == name]['avg_likelihood']
            if avg_likelihood.empty:
                updated_investor_score_dict[name] = score * 0
            else:
                updated_investor_score_dict[name] = score * avg_likelihood.values[0]
        sorted_dict = dict(sorted(updated_investor_score_dict.items(), key=lambda item: item[1], reverse=True))
        rerank_investors_100 = list(sorted_dict.keys())[:100]
        rerank_investors_10 = list(sorted_dict.keys())[:10]

    
    return recommended_investors_100,  rerank_investors_100, recommended_investors_10, rerank_investors_10 


# Example usage:
# Replace with your actual startup features
# df_new_startups = pd.read_csv('test_startups_r.csv')
 
# # 2. Clean the Data
# df_new_startups.drop('Unnamed: 0', inplace=True, axis=1)
# df_new_startups.drop('Unnamed: 0.1', inplace=True, axis=1)
# df_new_startups.fillna(0, inplace=True)
 
# num_columns = len(df_new_startups.columns)
# # print("Number of columns:", num_columns)
 
 
 
 
# print(df_new_startups[8:9]['org_uuid'])
# # X = df_new_startups.loc[df_new_startups['org_uuid'].drop_duplicates().index].drop(columns=['org_uuid','description', 'embeddings'])
# X = df_new_startups.loc[df_new_startups['org_uuid'].drop_duplicates().index].drop(columns=['org_uuid','description', 'embeddings'])
# X = X[8:9].values
# provided_startup_features = np.array(X)  # Replace with your actual features
# recommend(provided_startup_features)

  
# if __name__ == '__main__':
#     main()



from sentence_transformers import SentenceTransformer
def preprocess_startup_data(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    columns = ['org_uuid', '3D Printing', '3D Technology', 'A/B Testing', 'Accounting', 'Ad Exchange', 'Ad Network',
               'Ad Retargeting', 'Ad Server', 'Ad Targeting', 'Adult', 'Advanced Materials', 'Adventure Travel',
               'Advertising_x', 'Advertising Platforms', 'Advice', 'Aerospace', 'Affiliate Marketing', 'AgTech',
               'Agriculture', 'Air Transportation', 'Alternative Medicine', 'Alumni', 'American Football',
               'Amusement Park and Arcade', 'Analytics', 'Android', 'Angel Investment', 'Animal Feed', 'Animation',
               'App Discovery', 'App Marketing', 'Application Performance Management',
               'Application Specific Integrated Circuit (ASIC)', 'Apps_x', 'Aquaculture', 'Architecture',
               'Archiving Service', 'Art', 'Artificial Intelligence_x', 'Asset Management', 'Assisted Living',
               'Assistive Technology', 'Association', 'Auctions', 'Audio', 'Audiobooks', 'Augmented Reality',
               'Auto Insurance', 'Automotive', 'Autonomous Vehicles', 'B2B', 'B2C', 'Baby', 'Bakery', 'Banking',
               'Baseball', 'Basketball', 'Battery', 'Beauty', 'Big Data', 'Billing', 'Biofuel', 'Bioinformatics',
               'Biomass Energy', 'Biometrics', 'Biopharma', 'Biotechnology_x', 'Bitcoin', 'Blockchain',
               'Blogging Platforms', 'Boating', 'Brand Marketing', 'Brewing', 'Broadcasting', 'Browser Extensions',
               'Building Maintenance', 'Building Material', 'Business Development', 'Business Information Systems',
               'Business Intelligence', 'Business Travel', 'CAD', 'CMS', 'CRM', 'Call Center', 'Cannabis',
               'Car Sharing', 'Career Planning', 'Casino', 'Casual Games', 'Catering', 'Cause Marketing', 'Celebrity',
               'Charity', 'Charter Schools', 'Chemical', 'Chemical Engineering', 'Child Care', 'Children', 'CivicTech',
               'Civil Engineering', 'Classifieds', 'Clean Energy', 'CleanTech', 'Clinical Trials', 'Cloud Computing',
               'Cloud Data Services', 'Cloud Infrastructure', 'Cloud Management', 'Cloud Security', 'Cloud Storage',
               'Coffee', 'Collaboration', 'Collaborative Consumption', 'Collectibles', 'Collection Agency',
               'College Recruiting', 'Comics', 'Commercial', 'Commercial Insurance', 'Commercial Lending',
               'Commercial Real Estate', 'Communication Hardware', 'Communications Infrastructure', 'Communities',
               'Compliance', 'Computer', 'Computer Vision', 'Concerts', 'Confectionery', 'Console Games',
               'Construction', 'Consulting', 'Consumer', 'Consumer Applications', 'Consumer Electronics_x',
               'Consumer Goods_x', 'Consumer Lending', 'Consumer Research', 'Consumer Reviews', 'Consumer Software',
               'Contact Management', 'Content', 'Content Creators', 'Content Delivery Network', 'Content Discovery',
               'Content Marketing', 'Content Syndication', 'Contests', 'Continuing Education', 'Cooking',
               'Corporate Training', 'Corrections Facilities', 'Cosmetic Surgery', 'Cosmetics', 'Coupons',
               'Courier Service', 'Coworking', 'Craft Beer', 'Creative Agency', 'Credit', 'Credit Bureau',
               'Credit Cards', 'Cricket', 'Crowdfunding', 'Crowdsourcing', 'Cryptocurrency', 'Customer Service',
               'Cyber Security', 'Cycling', 'DIY', 'DRM', 'DSP', 'Darknet', 'Data Center', 'Data Center Automation',
               'Data Integration', 'Data Mining', 'Data Storage', 'Data Visualization', 'Database', 'Dating',
               'Debit Cards', 'Debt Collections', 'Delivery', 'Delivery Service', 'Dental', 'Desktop Apps',
               'Developer APIs', 'Developer Platform', 'Developer Tools', 'Diabetes', 'Dietary Supplements',
               'Digital Entertainment', 'Digital Marketing', 'Digital Media', 'Digital Signage', 'Direct Marketing',
               'Direct Sales', 'Distillery', 'Diving', 'Document Management', 'Document Preparation',
               'Domain Registrar', 'Drone Management', 'Drones', 'E-Commerce', 'E-Commerce Platforms', 'E-Learning',
               'E-Signature', 'EBooks', 'EdTech', 'Ediscovery', 'Education_x', 'Edutainment', 'Elder Care', 'Elderly',
               'Electric Vehicle', 'Electrical Distribution', 'Electronic Design Automation (EDA)',
               'Electronic Health Record (EHR)', 'Electronics', 'Email', 'Email Marketing', 'Embedded Software',
               'Embedded Systems', 'Emergency Medicine', 'Emerging Markets', 'Employee Benefits', 'Employment',
               'Energy_x', 'Energy Efficiency', 'Energy Management', 'Energy Storage', 'Enterprise',
               'Enterprise Applications', 'Enterprise Resource Planning (ERP)', 'Enterprise Software',
               'Environmental Consulting', 'Environmental Engineering', 'Equestrian', 'Ethereum', 'Event Management',
               'Event Promotion', 'Events_x', 'Extermination Service', 'Eyewear', 'Facebook', 'Facial Recognition',
               'Facilities Support Services', 'Facility Management', 'Family', 'Fantasy Sports', 'Farmers Market',
               'Farming', 'Fashion', 'Fast-Moving Consumer Goods', 'Ferry Service', 'Fertility', 'Field Support',
               'Field-Programmable Gate Array (FPGA)', 'File Sharing', 'Film', 'Film Distribution', 'Film Production',
               'FinTech', 'Finance', 'Financial Exchanges', 'Financial Services_x', 'First Aid', 'Fitness',
               'Flash Sale', 'Flash Storage', 'Fleet Management', 'Flowers', 'Food Delivery', 'Food Processing',
               'Food Trucks', 'Food and Beverage_x', 'Forestry', 'Fossil Fuels', 'Foundries', 'Franchise',
               'Fraud Detection', 'Freelance', 'Freemium', 'Freight Service', 'Fruit', 'Fuel', 'Fuel Cell',
               'Funding Platform', 'Funerals', 'Furniture', 'GPS', 'GPU', 'Gambling', 'Gamification', 'Gaming_x',
               'Generation Z', 'Genetics', 'Geospatial', 'Gift', 'Gift Card', 'Gift Exchange', 'Gift Registry', 'Golf',
               'Google', 'Google Glass', 'GovTech', 'Government', 'Graphic Design', 'Green Building',
               'Green Consumer Goods', 'GreenTech', 'Grocery', 'Group Buying', 'Guides', 'Handmade', 'Hardware_x',
               'Health Care_x', 'Health Diagnostics', 'Health Insurance', 'Hedge Funds', 'Higher Education', 'Hockey',
               'Home Decor', 'Home Health Care', 'Home Improvement', 'Home Renovation', 'Home Services',
               'Home and Garden', 'Homeland Security', 'Homeless Shelter', 'Horticulture', 'Hospital', 'Hospitality',
               'Hotel', 'Housekeeping Service', 'Human Computer Interaction', 'Human Resources', 'Humanitarian',
               'Hunting', 'Hydroponics', 'ISP', 'IT Infrastructure', 'IT Management', 'IaaS', 'Identity Management',
               'Image Recognition', 'Impact Investing', 'In-Flight Entertainment', 'Incubators', 'Independent Music',
               'Indoor Positioning', 'Industrial', 'Industrial Automation', 'Industrial Design',
               'Industrial Engineering', 'Industrial Manufacturing', 'Information Services', 'Information Technology_x',
               'Information and Communications Technology (ICT)', 'Infrastructure', 'Innovation Management',
               'InsurTech', 'Insurance', 'Intellectual Property', 'Intelligent Systems', 'Interior Design', 'Internet',
               'Internet Radio', 'Internet of Things', 'Intrusion Detection', 'Janitorial Service', 'Jewelry',
               'Journalism', 'Knowledge Management', 'LGBT', 'Landscaping', 'Language Learning', 'Laser',
               'Last Mile Transportation', 'Laundry and Dry-cleaning', 'Law Enforcement', 'Lead Generation',
               'Lead Management', 'Leasing', 'Legal', 'Legal Tech', 'Leisure', 'Lending', 'Life Insurance',
               'Life Science', 'Lifestyle', 'Lighting', 'Limousine Service', 'Lingerie', 'Linux', 'Livestock', 'Local',
               'Local Advertising', 'Local Business', 'Local Shopping', 'Location Based Services', 'Logistics',
               'Loyalty Programs', 'MMO Games', 'MOOC', 'Machine Learning', 'Machinery Manufacturing', 'Made to Order',
               'Management Consulting', 'Management Information Systems', 'Manufacturing_x', 'Mapping Services',
               'Marine Technology', 'Marine Transportation', 'Market Research', 'Marketing', 'Marketing Automation',
               'Marketplace', 'Mechanical Design', 'Mechanical Engineering', 'Media and Entertainment_x', 'Medical',
               'Medical Device', 'Meeting Software', "Men's", 'Messaging', 'Micro Lending', 'Military', 'Millennials',
               'Mineral', 'Mining', 'Mining Technology', 'Mobile_x', 'Mobile Advertising', 'Mobile Apps',
               'Mobile Devices', 'Mobile Payments', 'Motion Capture', 'Multi-level Marketing',
               'Museums and Historical Sites', 'Music', 'Music Education', 'Music Label', 'Music Streaming',
               'Music Venues', 'Musical Instruments', 'NFC', 'Nanotechnology', 'National Security',
               'Natural Language Processing', 'Natural Resources_x', 'Navigation', 'Network Hardware',
               'Network Security', 'Neuroscience', 'News', 'Nightclubs', 'Nightlife', 'Non Profit', 'Nuclear',
               'Nursing and Residential Care', 'Nutraceutical', 'Nutrition', 'Office Administration', 'Oil and Gas',
               'Online Auctions', 'Online Forums', 'Online Games', 'Online Portals', 'Open Source', 'Operating Systems',
               'Optical Communication', 'Organic', 'Organic Food', 'Outdoor Advertising', 'Outdoors', 'Outpatient Care',
               'Outsourcing', 'PC Games', 'PaaS', 'Packaging Services', 'Paper Manufacturing', 'Parenting', 'Parking',
               'Parks', 'Payments_x', 'Peer to Peer', 'Penetration Testing', 'Performing Arts', 'Personal Branding',
               'Personal Development', 'Personal Finance', 'Personal Health', 'Personalization', 'Pet',
               'Pharmaceutical', 'Photo Editing', 'Photo Sharing', 'Photography', 'Physical Security',
               'Plastics and Rubber Manufacturing', 'Playstation', 'Podcast', 'Point of Sale', 'Politics',
               'Pollution Control', 'Ports and Harbors', 'Power Grid', 'Precious Metals', 'Prediction Markets',
               'Predictive Analytics', 'Presentation Software', 'Presentations', 'Price Comparison',
               'Primary Education', 'Printing', 'Privacy', 'Private Cloud', 'Private Social Networking', 'Procurement',
               'Product Design', 'Product Management', 'Product Research', 'Product Search', 'Productivity Tools',
               'Professional Networking', 'Professional Services_x', 'Project Management', 'Property Development',
               'Property Insurance', 'Property Management', 'Psychology', 'Public Relations', 'Public Safety',
               'Public Transportation', 'Publishing', 'Q&A', 'QR Codes', 'Quality Assurance', 'Quantified Self',
               'Quantum Computing', 'RFID', 'Racing', 'Railroad', 'Reading Apps', 'Real Estate_x',
               'Real Estate Investment', 'Real Time', 'Recipes', 'Recreation', 'Recreational Vehicles', 'Recruiting',
               'Recycling', 'Rehabilitation', 'Religion', 'Renewable Energy', 'Rental', 'Rental Property', 'Reputation',
               'Reservations', 'Residential', 'Resorts', 'Restaurants', 'Retail', 'Retail Technology', 'Retirement',
               'Ride Sharing', 'Risk Management', 'Robotics', 'Roku', 'SEM', 'SEO', 'SMS', 'SNS', 'STEM Education',
               'SaaS', 'Sailing', 'Sales', 'Sales Automation', 'Same Day Delivery', 'Satellite Communication',
               'Scheduling', 'Seafood', 'Search Engine', 'Secondary Education', 'Security', 'Self-Storage',
               'Semantic Search', 'Semantic Web', 'Semiconductor', 'Sensor', 'Serious Games', 'Service Industry',
               'Sex Industry', 'Sex Tech', 'Sharing Economy', 'Shipping', 'Shipping Broker', 'Shoes', 'Shopping',
               'Shopping Mall', 'Simulation', 'Skiing', 'Skill Assessment', 'Small and Medium Businesses',
               'Smart Building', 'Smart Cities', 'Smart Home', 'Snack Food', 'Soccer', 'Social', 'Social Assistance',
               'Social Bookmarking', 'Social CRM', 'Social Entrepreneurship', 'Social Impact', 'Social Media',
               'Social Media Advertising', 'Social Media Management', 'Social Media Marketing', 'Social Network',
               'Social News', 'Social Recruiting', 'Social Shopping', 'Software_x', 'Software Engineering', 'Solar',
               'Space Travel', 'Spam Filtering', 'Speech Recognition', 'Sponsorship', 'Sporting Goods', 'Sports_x',
               'Staffing Agency', 'Stock Exchanges', 'Subscription Service', 'Supply Chain Management', 'Surfing',
               'Sustainability_x', 'Swimming', 'TV', 'TV Production', 'Table Tennis', 'Task Management', 'Taxi Service',
               'Tea', 'Technical Support', 'Teenagers', 'Telecommunications', 'Tennis', 'Test and Measurement',
               'Text Analytics', 'Textbook', 'Textiles', 'Theatre', 'Therapeutics', 'Ticketing', 'Timber', 'Timeshare',
               'Tobacco', 'Tour Operator', 'Tourism', 'Toys', 'Trade Shows', 'Trading Platform', 'Training',
               'Transaction Processing', 'Translation Service', 'Transportation_x', 'Travel', 'Travel Accommodations',
               'Travel Agency', 'Tutoring', 'Twitter', 'UX Design', 'Underserved Children', 'Unified Communications',
               'Universities', 'Usability Testing', 'Vacation Rental', 'Vending and Concessions', 'Venture Capital',
               'Vertical Search', 'Veterinary', 'Video_x', 'Video Advertising', 'Video Chat', 'Video Conferencing',
               'Video Editing', 'Video Games', 'Video Streaming', 'Video on Demand', 'Virtual Assistant',
               'Virtual Currency', 'Virtual Desktop', 'Virtual Goods', 'Virtual Reality', 'Virtual Workforce',
               'Virtual World', 'Virtualization', 'Visual Search', 'VoIP', 'Vocational Education', 'Volley Ball',
               'Warehousing', 'Waste Management', 'Water', 'Water Purification', 'Water Transportation',
               'Wealth Management', 'Wearables', 'Web Apps', 'Web Browsers', 'Web Design', 'Web Development',
               'Web Hosting', 'WebOS', 'Wedding', 'Wellness', 'Wholesale', 'Wind Energy', 'Windows', 'Windows Phone',
               'Wine And Spirits', 'Winery', 'Wired Telecommunications', 'Wireless', "Women's", 'Wood Processing',
               'Xbox', 'Young Adults', 'eSports', 'iOS', 'mHealth', 'macOS', 'Administrative Services', 'Advertising_y',
               'Agriculture and Farming', 'Apps_y', 'Artificial Intelligence_y', 'Biotechnology_y',
               'Clothing and Apparel', 'Commerce and Shopping', 'Community and Lifestyle', 'Consumer Electronics_y',
               'Consumer Goods_y', 'Content and Publishing', 'Data and Analytics', 'Design', 'Education_y', 'Energy_y',
               'Events_y', 'Financial Services_y', 'Food and Beverage_y', 'Gaming_y', 'Government and Military',
               'Hardware_y', 'Health Care_y', 'Information Technology_y', 'Internet Services',
               'Lending and Investments', 'Manufacturing_y', 'Media and Entertainment_y',
               'Messaging and Telecommunications', 'Mobile_y', 'Music and Audio', 'Natural Resources_y',
               'Navigation and Mapping', 'Payments_y', 'Platforms', 'Privacy and Security', 'Professional Services_y',
               'Real Estate_y', 'Sales and Marketing', 'Science and Engineering', 'Software_y', 'Sports_y',
               'Sustainability_y', 'Transportation_y', 'Travel and Tourism', 'Video_y', 'AFG', 'AGO', 'AIA', 'ALB',
               'ARE', 'ARG', 'ARM', 'ASM', 'AUS', 'AUT', 'AZE', 'BAH', 'BDI', 'BEL', 'BEN', 'BFA', 'BGD', 'BGR', 'BHR',
               'BIH', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'BRN', 'BWA', 'CAN', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR',
               'COD', 'COL', 'COM', 'CPV', 'CRI', 'CYM', 'CYP', 'CZE', 'DEU', 'DJI', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY',
               'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FRA', 'GAB', 'GBR', 'GEO', 'GGY', 'GHA', 'GIB', 'GIN', 'GMB', 'GRC',
               'GRD', 'GRL', 'GTM', 'HKG', 'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IMN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL',
               'ISR', 'ITA', 'JAM', 'JEY', 'JOR', 'JPN', 'KAZ', 'KEN', 'KHM', 'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR',
               'LIE', 'LKA', 'LSO', 'LTU', 'LUX', 'LVA', 'MAC', 'MAF', 'MAR', 'MDA', 'MDG', 'MEX', 'MKD', 'MLI', 'MLT',
               'MMR', 'MNG', 'MOZ', 'MRT', 'MTQ', 'MUS', 'MWI', 'MYS', 'NAM', 'NER', 'NGA', 'NIC', 'NLD', 'NOR', 'NPL',
               'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PNG', 'POL', 'PRI', 'PRT', 'PRY', 'PSE', 'QAT', 'ROM', 'RUS',
               'RWA', 'SAU', 'SEN', 'SGP', 'SLE', 'SLV', 'SRB', 'SVK', 'SVN', 'SWE', 'SWZ', 'SYC', 'TAN', 'TGO', 'THA',
               'TTO', 'TUN', 'TUR', 'TWN', 'UGA', 'UKR', 'URY', 'USA', 'VEN', 'VNM', 'ZAF', 'ZMB', 'ZWE', 'Aargau',
               'Aberdeen City', 'Aberdeenshire', 'Abruzzi', 'Abu Dhabi', 'Acre', 'Ad Dawhah', 'Adis Abeba',
               'Aguascalientes', 'Aichi', 'Ain Defla', 'Aisen del General Carlos Ibanez del Campo', 'Aizkraukles',
               'Ajdovscina Commune', 'Ajman', 'Akershus', 'Akhaia', 'Akita', "Al Balqa'", 'Al Buhayrah',
               'Al Iskandariyah', "Al Isma'iliyah", 'Al Jizah', 'Al Kuwayt', 'Al Manamah', 'Al Mintaqah ash Shamaliyah',
               'Al Munastir', 'Al Qahirah', 'Alabama', 'Alagoas', 'Alajuela', 'Aland', 'Alaska', 'Alba', 'Alberta',
               'Alger', 'Almaty', 'Almaty City', 'Alsace', 'Alytaus Apskritis', 'Amara', 'Amazonas', 'Amman',
               'Andalucia', 'Andhra Pradesh', 'Angus', 'Anhui', 'Anjouan', 'Ankara', 'Antalya', 'Antananarivo',
               'Antioquia', 'Antrim', 'Antwerpen', 'Aomori', 'Aquitaine', 'Ar Riyad', 'Aragon', 'Araucania', 'Arbil',
               'Ards', 'Arequipa', 'Argyll and Bute', 'Arizona', 'Arkansas', 'Armagh', 'Arnessysla', 'Arta', 'Arusha',
               'As Suways', 'Ash Sharqiyah', 'Ashanti', 'Assam', 'Astana', "Astrakhan'", 'Asturias', 'Asyut',
               'Atlantico', 'Attiki', 'Auckland', 'Ausser-Rhoden', 'Aust-Agder', 'Australian Capital Territory',
               'Austur-Skaftafellssysla', 'Austurland', 'Auvergne', 'Aveiro', 'Azores', 'Azuay', 'Bacs-Kiskun',
               'Baden-Wurttemberg', 'Bagmati', 'Baguio', 'Bahia', 'Baja California', 'Baki', 'Bali', 'Balikesir',
               'Ballymena', 'Bamako', 'Banjul', 'Banska Bystrica', 'Banten', 'Barisal', 'Barnet', 'Barnsley',
               'Basel-Landschaft', 'Basel-Stadt', 'Bashkortostan', 'Basilicata', 'Basse-Normandie',
               'Bath and North East Somerset', 'Bauskas', 'Bay of Plenty', 'Bayern', 'Beau Vallon', 'Bedfordshire',
               'Beijing', 'Beja', 'Bejaia', 'Belfast', 'Belgorod', 'Belize', 'Benguet', 'Berlin', 'Bern', 'Beyrouth',
               'Bihar', 'Bihor', 'Birmingham', 'Bjelovarsko-Bilogorska', 'Black River', 'Blackburn with Darwen',
               'Blackpool', 'Blaenau Gwent', 'Blantyre', 'Blekinge Lan', 'Bohinj Commune', 'Bolton',
               'Borsod-Abauj-Zemplen', 'Boumerdes', 'Bourgogne', 'Bournemouth', 'Brabant Wallon', 'Bracknell Forest',
               'Bradford', 'Braga', 'Braganca', 'Brandenburg', 'Brasov', 'Bratislava', 'Bremen', 'Brent',
               "Brestskaya Voblasts'", 'Bretagne', 'Brezovica Commune', 'Bridgend', 'Brighton and Hove',
               'Bristol, City of', 'British Columbia', 'Bromley', 'Brong-Ahafo', 'Brunei and Muara',
               'Brussels Hoofdstedelijk Gewest', 'Buckinghamshire', 'Bucuresti', 'Budapest', 'Buenos Aires',
               'Bujumbura', 'Burgas', 'Burgenland', 'Bursa', 'Bury', 'Buryat', 'Buskerud', 'Butare', 'Caerphilly',
               'Cagayan', 'Calabria', 'Caldas', 'Calderdale', 'California', 'Cambridgeshire', 'Camden', 'Campania',
               'Campeche', 'Canakkale', 'Canarias', 'Cantabria', 'Canterbury', 'Caras-Severin', 'Cardiff', 'Carlow',
               'Carmarthenshire', 'Cartago', 'Casanare', 'Castelo Branco', 'Castilla y Leon', 'Castilla-La Mancha',
               'Catalonia', 'Cavan', 'Cavite', 'Ceara', 'Cebu City', 'Central', 'Central Region', 'Centre',
               'Ceredigion', 'Cesu', "Ch'ungch'ong-bukto", "Ch'ungch'ong-namdo", 'Chaco', 'Champagne-Ardenne',
               'Chandigarh', 'Chaouia-Ouardigha', 'Cheju-do', 'Chelyabinsk', "Chernivets'ka Oblast'", 'Cheshire',
               'Cheshire East', 'Cheshire West and Chester', 'Chhattisgarh', 'Chiang Mai', 'Chiapas', 'Chiba',
               'Chihuahua', 'Chisinau', 'Cholla-bukto', 'Cholla-namdo', 'Chon Buri', 'Chongqing', 'Christ Church',
               'Christ Church Nichola Town', 'Chukot', 'Chuvashia', 'Clackmannanshire', 'Clare', 'Cluj',
               'Coahuila de Zaragoza', 'Coast', 'Cocle', 'Coimbra', 'Coleraine', 'Colima', 'Colorado',
               'Comunidad Valenciana', 'Connecticut', 'Conwy', 'Cookstown', 'Cordoba', 'Cork', 'Cornwall', 'Corse',
               'Cortes', 'Coventry', 'Craigavon', 'Cross River', 'Croydon', 'Csongrad', 'Cumbria', 'Cundinamarca',
               'Cusco', 'Dakar', 'Dalarnas Lan', 'Dambovita', 'Dar es Salaam', 'Davao City', 'Debrecen', 'Delaware',
               'Delhi', 'Denbighshire', 'Denizli', 'Derby', 'Derbyshire', 'Derry', 'Devon', 'Dhaka', 'Diekirch',
               'Dire Dawa', 'District of Columbia', 'Distrito Especial', 'Distrito Federal', 'Distrito Nacional',
               "Dnipropetrovs'ka Oblast'", 'Dolj', 'Dolnoslaskie', 'Doncaster', 'Donegal', 'Dorset', 'Down', 'Drenthe',
               'Dubai', 'Dublin', 'Dubrovacko-Neretvanska', 'Dudley', 'Dundee City', 'Durango', 'Durham',
               "Dushet'is Raioni", 'Ealing', 'East Lothian', 'East Renfrewshire', 'East Riding of Yorkshire',
               'East Sussex', 'Eastern', 'Eastern Finland', 'Ebonyi', 'Edinburgh, City of', 'Edirne', 'Edo', 'Ehime',
               'Ekiti', 'Emilia-Romagna', 'Enfield', 'England', 'Entre Rios', 'Enugu', 'Eskisehir', 'Espirito Santo',
               'Essex', 'Esteli', 'Estuaire', 'Evora', 'Evvoia', 'Extremadura', 'Eyjafjardarsysla', 'Falkirk', 'Faro',
               'Fars', 'Federal Capital Territory', 'Federally Administered Tribal Areas',
               'Federation of Bosnia and Herzegovina', 'Fejer', 'Fermanagh', 'Fes-Boulemane', 'Fife', 'Flevoland',
               'Flintshire', 'Florida', 'Franche-Comte', 'Francisco Morazan', 'Freeport', 'Fribourg', 'Friesland',
               'Friuli-Venezia Giulia', 'Fujairah', 'Fujian', 'Fukui', 'Fukuoka', 'Fukushima', 'Gabrovo', 'Galicia',
               'Galway', 'Gansu', 'Gateshead', 'Gavleborgs Lan', 'Gaza', 'Gaziantep', 'Gelderland', 'Geneve', 'Georgia',
               'Gibraltar', 'Gifu', 'Gisborne', 'Gitarama', 'Gjirokaster', 'Glarus', 'Glasgow City', 'Gloucestershire',
               'Goa', 'Goias', 'Gorenja vas-Poljane Commune', 'Gornja Radgona Commune', 'Gotlands Lan', 'Grad Sofiya',
               'Grad Zagreb', 'Grand Casablanca', 'Graubunden', 'Greater Accra', 'Greenwich', 'Grevenmacher',
               'Groningen', 'Grosuplje Commune', 'Guanacaste', 'Guanajuato', 'Guangdong', 'Guangxi', 'Guarda',
               'Guatemala', 'Guayas', 'Guerrero', 'Guizhou', 'Gujarat', 'Gullbringusysla', 'Gumma', 'Gwynedd', 'Gyor',
               'Gyor-Moson-Sopron', "Ha'il", 'HaDarom', 'HaMerkaz', 'HaZafon', 'Hackney', 'Hainan', 'Hainaut',
               'Hallands Lan', 'Halton', 'Hamburg', 'Hamilton', 'Hammersmith and Fulham', 'Hampshire', 'Harbour Island',
               'Harjumaa', 'Harrow', 'Hartlepool', 'Haryana', 'Hatay', 'Haute-Normandie', 'Havering', 'Hawaii',
               "Hawke's Bay", 'Hebei', 'Hedmark', 'Hefa', 'Heilongjiang', 'Henan', 'Herefordshire', 'Hertford',
               'Hessen', 'Heves', 'Hhohho', 'Hidalgo', 'Highland', 'Hiiumaa', 'Hillingdon', 'Himachal Pradesh',
               'Hiroshima', 'Hlavni mesto Praha', 'Hokkaido', 'Hong Kong Island', 'Hordaland', 'Houet', 'Hounslow',
               'Hovedstaden', 'Hrpelje-Kozina Commune', 'Hubei', 'Huehuetenango', 'Huila', 'Hunan', 'Hyogo', 'Iasi',
               'Ibaraki', 'Idaho', 'Idrija Commune', 'Ile-de-France', 'Ilfov', 'Ilhas', 'Illinois', 'Imo',
               "Inch'on-jikhalsi", 'Indiana', 'Inner-Rhoden', 'Inverclyde', 'Ioannina', 'Iowa', 'Iraklion', 'Iringa',
               'Irkutsk', 'Isabela', 'Islamabad', 'Islas Baleares', 'Isle of Anglesey', 'Isle of Wight', 'Islington',
               'Isparta', 'Istanbul', 'Istarska', 'Ivanovo', 'Iwate', 'Izmir', 'Jakarta Raya', 'Jalisco',
               'Jammu and Kashmir', 'Jamtlands Lan', "Janub Sina'", 'Jasz-Nagykun-Szolnok', 'Jawa Barat', 'Jawa Tengah',
               'Jawa Timur', 'Jelgavas', 'Jharkhand', 'Jiangsu', 'Jiangxi', 'Jihocesky kraj', 'Jihomoravsky kraj',
               'Jilin', 'Jinja', 'Jizan', 'Johor', 'Jonkopings Lan', 'Jujuy', 'Jura', 'Kabol', 'Kadiogo', 'Kaduna',
               'Kagoshima', 'Kalimantan Barat', 'Kaliningrad', 'Kalmar Lan', 'Kampala', 'Kampot', 'Kanagawa',
               'Kangwon-do', 'Kankan', 'Kano', 'Kansas', 'Karlovarsky kraj', 'Karnataka', 'Karnten', 'Karpos',
               'Katanga', 'Katsina', 'Kauno Apskritis', 'Kavala', 'Kayes', 'Kayseri', 'Kedah', 'Kelantan', 'Kemerovo',
               'Kent', 'Kentucky', 'Kepulauan Riau', 'Kerala', 'Kerry', 'Khabarovsk', 'Khanty-Mansiy',
               "Kharkivs'ka Oblast'", 'Khorasan', 'Kigali', 'Kildare', 'Kilimanjaro', 'Kilkenny', 'Kilkis', 'Kingston',
               'Kingston upon Hull, City of', 'Kingston upon Thames', 'Kinshasa', 'Kirklees', "Kirovohrads'ka Oblast'",
               'Klaipedos Apskritis', 'Knowsley', 'Kocaeli', 'Kochi', 'Kogi', 'Komarom-Esztergom', 'Komi', 'Konya',
               'Kosice', 'Kowloon', 'Krabi', 'Kralovehradecky kraj', 'Kranj Commune', 'Krapinsko-Zagorska', 'Krasnodar',
               'Krasnoyarsk', 'Krasnoyarskiy Kray', 'Kronobergs Lan', 'Krsko Commune', 'Krung Thep', 'Kuala Lumpur',
               'Kujawsko-Pomorskie', 'Kumamoto', 'Kwangju-jikhalsi', 'Kwara', 'Kyonggi-do', 'Kyongsang-bukto',
               'Kyongsang-namdo', 'Kyoto', 'Kyyiv', "Kyyivs'ka Oblast'", "L'vivs'ka Oblast'", 'La Libertad', 'La Paz',
               'La Rioja', 'La Union', 'Laanemaa', 'Lagos', 'Lagunes', 'Lambeth', 'Lancashire', 'Languedoc-Roussillon',
               'Laoag', 'Laois', 'Lapland', 'Larnaca', 'Larne', 'Lasithi', 'Lazio', 'Leeds', 'Legaspi', 'Leicester',
               'Leicestershire', 'Leiria', 'Liaoning', 'Liban-Nord', 'Liberecky kraj', 'Licko-Senjska', 'Liege',
               'Liguria', 'Lilongwe', 'Lima', 'Limassol', 'Limbazu', 'Limburg', 'Limerick', 'Limousin', 'Lincolnshire',
               'Lisboa', 'Lisburn', 'Littoral', 'Liverpool', 'Ljubljana Urban Commune', 'Ljutomer Commune', 'Lodzkie',
               'Loei', 'Loja', 'Lombardia', 'Longford', 'Lorraine', 'Los Lagos', 'Louisiana', 'Louth', 'Luanda',
               'Lubelskie', 'Lubuskie', 'Luxembourg', 'Luzern', 'Madang', 'Madeira', 'Madhya Pradesh', 'Madrid',
               'Magallanes y de la Antartica Chilena', 'Magdalena', 'Maharashtra', 'Maine', 'Makkah', 'Malatya',
               'Malopolskie', 'Manabi', 'Managua', 'Manchester', 'Mandalay', 'Manica', 'Manila', 'Manisa', 'Manitoba',
               'Maputo', 'Maramures', 'Maranhao', 'Marche', 'Mardin', 'Maribor Commune', 'Maritime', 'Marlborough',
               'Marrakech-Tensift-Al Haouz', 'Maryland', 'Masaka', 'Maseru', 'Masqat', 'Massachusetts', 'Mato Grosso',
               'Mato Grosso do Sul', 'Maule', 'Mayo', 'Mazowieckie', 'Mbarara', 'Mbeya', 'Meath',
               'Mecklenburg-Vorpommern', 'Medway', 'Melaka', 'Mendoza', 'Mersin', 'Merton', 'Mexico', 'Michigan',
               'Michoacan de Ocampo', 'Middlesbrough', 'Midi-Pyrenees', 'Midland', 'Midlothian', 'Midtjylland',
               'Milton Keynes', 'Minas Gerais', 'Minnesota', 'Minsk', "Minskaya Voblasts'", 'Miskolc', 'Mississippi',
               'Missouri', 'Miyagi', 'Miyazaki', 'Moka', 'Molise', 'Monaghan', 'Monmouthshire', 'Mont-Liban', 'Montana',
               'Montserrado', 'Moravskoslezsky kraj', 'Moray', 'Mordovia', 'More og Romsdal', 'Morelos', 'Morobe',
               'Moscow City', 'Moskva', 'Mugla', 'Murcia', 'Mures', 'NA - American Samoa', 'NA - Anguilla',
               'NA - Guernsey', 'NA - Isle of Man', 'NA - Jersey', 'NA - Malta', 'NA - Martinique', 'NA - Puerto Rico',
               'NA - Saint-Martin (France)', 'NA - South Africa', 'NA - Uruguay', 'NA - Venezuela', 'NA - Vietnam',
               'NA - Zambia', 'NA - Zimbabwe', 'Nabeul', 'Nagano', 'Nagasaki', 'Nairobi Area', 'Nampula', 'Namur',
               'Narino', 'Nassarawa', 'Navarra', 'Neath Port Talbot', 'Nebraska', 'Negeri Sembilan',
               'Negros Occidental', 'Negros Oriental', 'Nei Mongol', 'Nelson', 'Neuchatel', 'Neuquen', 'Nevada',
               'New Brunswick', 'New Hampshire', 'New Jersey', 'New Mexico', 'New Providence', 'New South Wales',
               'New Territories', 'New York', 'Newcastle upon Tyne', 'Newfoundland', 'Newport', 'Newry and Mourne',
               'Niamey', 'Nicosia', 'Nidwalden', 'Niederosterreich', 'Niedersachsen', 'Niigata', 'Nitra', 'Nizhegorod',
               'Nograd', 'Nonthaburi', 'Noord-Brabant', 'Noord-Holland', 'Nord', 'Nord-Kivu', 'Nord-Pas-de-Calais',
               'Nord-Trondelag', 'Nordjylland', 'Nordland', 'Nordrhein-Westfalen', 'Norfolk', 'Norourland Eystra',
               'Norrbottens Lan', 'North Ayrshire', 'North Carolina', 'North Dakota', 'North East Lincolnshire',
               'North Lanarkshire', 'North Lincolnshire', 'North Region', 'North Somerset', 'North Tyneside',
               'North Yorkshire', 'North-West Frontier', 'Northamptonshire', 'Northern', 'Northumberland', 'Nottingham',
               'Nottinghamshire', 'Nova Scotia', 'Novgorod', 'Novosibirsk', 'Nuevo Leon', 'Nunavut', 'Nyanza', 'Oaxaca',
               'Oberosterreich', 'Obwalden', "Odes'ka Oblast'", 'Offaly', 'Ogun', 'Ohio', 'Okayama', 'Okinawa',
               'Oklahoma', 'Oldham', 'Olomoucky kraj', 'Omagh', 'Omsk', 'Ondo', 'Ontario', 'Oost-Vlaanderen',
               'Opolskie', 'Oppland', 'Orebro Lan', 'Oregon', 'Orenburg', 'Orissa', 'Orkney', 'Oromiya', 'Osaka',
               'Oshana', 'Osjecko-Baranjska', 'Oslo', 'Osmaniye', 'Ostergotlands Lan', 'Ostfold', 'Osun', 'Otago',
               'Otjozondjupa', 'Ouest', 'Oulu', 'Overijssel', 'Oxfordshire', 'Oyo', 'Pais Vasco', 'Pampanga',
               'Pamplemousses', 'Panama', 'Pangasinan', 'Paphos', 'Para', 'Paraiba', 'Parana', 'Pardubicky kraj',
               'Parnumaa', 'Pasay', 'Pathum Thani', 'Pays de la Loire', 'Pecs', 'Pembrokeshire', 'Pennsylvania',
               'Perak', 'Permskiy Kray', 'Pernambuco', 'Perth and Kinross', 'Pesnica Commune', 'Pest', 'Peterborough',
               'Phitsanulok', 'Phnum Penh', 'Phuket', 'Piaui', 'Picardie', 'Pichincha', 'Piemonte', 'Piura',
               'Plaines Wilhems', 'Plateau', 'Plovdiv', 'Plymouth', 'Plzensky kraj', 'Podkarpackie', 'Podlaskie',
               'Poitou-Charentes', 'Pomorskie', 'Port Louis', 'Port-of-Spain', 'Porto', 'Powys', 'Pozesko-Slavonska',
               'Prahova', 'Presov', 'Prilep', 'Prince Edward Island', "Provence-Alpes-Cote d'Azur", 'Puducherry',
               'Puebla', 'Puerto Plata', 'Puglia', 'Pulau Pinang', 'Punjab', 'Pusan-jikhalsi', 'Qina', 'Qinghai',
               'Quebec', 'Queensland', 'Queretaro de Arteaga', 'Quetzaltenango', 'Quezon City', 'Quintana Roo',
               'Rabat-Sale-Zemmour-Zaer', 'Radovljica Commune', 'Rajasthan', 'Rangarvallasysla', 'Rangoon', 'Raplamaa',
               'Ras Al Khaimah', 'Rayong', 'Reading', 'Redbridge', 'Redcar and Cleveland', 'Region Metropolitana',
               'Renfrewshire', 'Republika Srpska', 'Rezeknes', 'Rheinland-Pfalz', 'Rhode Island', 'Rhondda Cynon Taff',
               'Rhone-Alpes', 'Richmond upon Thames', 'Rift Valley', 'Riga', 'Rigas', 'Rio Grande do Norte',
               'Rio Grande do Sul', 'Rio de Janeiro', 'Rivers', 'Riviere du Rempart', 'Rizal', 'Rochdale', 'Rodhopi',
               'Rogaland', 'Rondonia', 'Roscommon', 'Rostov', 'Rotherham', 'Ruggell', 'Rutland', 'Saarland', 'Sabah',
               'Sachsen', 'Sachsen-Anhalt', 'Saga', 'Saint Andrew', 'Saint George', 'Saint John Figtree',
               'Saint Michael', 'Saint Patrick', 'Saint Petersburg City', 'Saint-Louis', 'Saitama', 'Sakha', 'Saldus',
               'Salford', 'Salzburg', 'Samara', 'Samut Prakan', 'Samut Sakhon', 'San Jose', 'San Luis Potosi',
               'San Salvador', 'San Vicente', 'Sandwell', 'Sankt Gallen', 'Sanliurfa', 'Santa Ana', 'Santa Catarina',
               'Santa Fe', 'Santarem', 'Santiago del Estero', 'Sao Domingos', 'Sao Paulo', 'Saratov', 'Sardegna',
               'Saskatchewan', 'Schaffhausen', 'Schleswig-Holstein', 'Schwyz', 'Scottish Borders, The', 'Sefton',
               'Selangor', 'Sentjur pri Celju Commune', "Seoul-t'ukpyolsi", 'Sergipe', 'Setubal', 'Sfax', 'Shaanxi',
               'Shandong', 'Shanghai', 'Shanxi', 'Sharjah', 'Sheffield', 'Shimane', 'Shizuoka', 'Shropshire',
               'Siauliu Apskritis', 'Sibiu', 'Sichuan', 'Sicilia', 'Sidi Bou Zid', 'Sikkim', 'Sinaloa', 'Sindh',
               'Sinop', 'Sitrah', 'Sjelland', 'Skagafjardarsysla', 'Skane Lan', 'Slaskie', 'Sligo', 'Slough',
               'Slovenska Bistrica Commune', 'Smolensk', 'Sodermanlands Lan', 'Sofala', 'Sogn og Fjordane', 'Solihull',
               'Solothurn', 'Somerset', 'Somogy', 'Songkhla', 'Sonora', 'Sor-Trondelag', 'Souss-Massa-Dr,a', 'Sousse',
               'South Australia', 'South Ayrshire', 'South Carolina', 'South Dakota', 'South Gloucestershire',
               'South Lanarkshire', 'South Tyneside', 'South-East', 'Southampton', 'Southend-on-Sea', 'Southern',
               'Southern Finland', 'Southwark', 'Splitsko-Dalmatinska', 'St. Helens', 'Staffordshire', 'Stann Creek',
               "Stavropol'", 'Steiermark', 'Stirling', 'Stockholms Lan', 'Stockport', 'Stockton-on-Tees',
               'Stoke-on-Trent', 'Stredocesky kraj', 'Struga', 'Strumica', 'Suchitepequez', 'Sud-Ouest',
               'Sudur-Mulasysla', 'Sudur-Tingeyjarsysla', 'Suffolk', 'Sulawesi Selatan', 'Sulawesi Tenggara',
               'Sumatera Utara', "Sums'ka Oblast'", 'Sunderland', 'Suourland', 'Suournes', 'Surrey', 'Sutton',
               'Sverdlovsk', 'Swansea', 'Swietokrzyskie', 'Syddanmark', 'Szabolcs-Szatmar-Bereg', 'Szeged', "T'ai-pei",
               "T'ai-wan", "T'bilisi", 'Tabasco', 'Taegu-jikhalsi', 'Taejon-jikhalsi', 'Takeo', 'Talsu', 'Tamaulipas',
               'Tambovskaya oblast', 'Tameside', 'Tamil Nadu', 'Tanga', 'Tanger-Tetouan', 'Taranaki', 'Tartumaa',
               'Tasmania', 'Tatarstan', 'Tehran', 'Tel Aviv', 'Telemark', 'Telford and Wrekin', 'Tennessee', 'Tete',
               'Texas', 'Thessaloniki', 'Thurgau', 'Thuringen', 'Thurrock', 'Tianjin', 'Ticino', 'Timis', 'Tipperary',
               'Tirane', 'Tirol', 'Toamasina', 'Tocantins', 'Tokushima', 'Tokyo', 'Toliara', 'Tolima', 'Tolna', 'Tomsk',
               'Torbay', 'Toscana', 'Toyama', 'Trafford', 'Trarza', 'Trbovlje Commune', 'Trentino-Alto Adige',
               'Triesen', 'Trikala', 'Trnava', 'Troms', 'Tucuman', 'Tukuma', 'Tunis', "Tver'", "Tyumen'", 'Udmurt',
               'Ulaanbaatar', 'Umbria', 'Umm Al Quwain', 'Uppsala Lan', 'Usak', 'Ustecky kraj', 'Usulutan', 'Utah',
               'Utenos Apskritis', 'Utrecht', 'Uttar Pradesh', 'Uttarakhand', 'Vaduz', 'Valais', 'Valcea',
               'Vale of Glamorgan, The', "Valle d'Aosta", 'Valle del Cauca', 'Valmieras', 'Valparaiso', 'Varmlands Lan',
               'Varna', 'Vas', 'Vasterbottens Lan', 'Vasternorrlands Lan', 'Vastmanlands Lan', 'Vastra Gotaland',
               'Vaud', 'Velenje Urban Commune', 'Veneto', 'Ventspils', 'Veracruz-Llave', 'Vermont', 'Vest-Agder',
               'Vestfiroir', 'Vestfold', 'Vestgronland', 'Vestur-Isafjardarsysla', 'Vesturland', 'Veszprem',
               'Viana do Castelo', 'Victoria', 'Vientiane', 'Vila Real', 'Viljandimaa', 'Vilniaus Apskritis',
               "Vinnyts'ka Oblast'", 'Virginia', 'Viseu', 'Vlaams-Brabant', 'Vojvodina', 'Volgograd', 'Volta',
               'Vorarlberg', 'Voronezh', 'Vukovarsko-Srijemska', 'Waikato', 'Wakayama', 'Wakefield', 'Walsall',
               'Waltham Forest', 'Wandsworth', 'Warminsko-Mazurskie', 'Warrington', 'Warwickshire', 'Washington',
               'Waterford', 'Wellington', 'West Bank', 'West Bengal', 'West Berkshire', 'West Dunbartonshire',
               'West Kazakhstan', 'West Lothian', 'West Region', 'West Sussex', 'West Virginia', 'West-Vlaanderen',
               'Western', 'Western Area', 'Western Australia', 'Western Finland', 'Westmeath', 'Westminster', 'Wexford',
               'Wicklow', 'Wielkopolskie', 'Wien', 'Wigan', 'Wiltshire', 'Windhoek', 'Windsor and Maidenhead', 'Wirral',
               'Wisconsin', 'Wokingham', 'Wolverhampton', 'Worcestershire', 'Wrexham', 'Wyoming', 'Xinjiang', 'Xizang',
               'Yalova', 'Yamagata', 'Yamaguchi', 'Yamanashi', 'Yangon', "Yaroslavl'", 'Yerevan', 'Yerushalayim',
               'Yogyakarta', 'York', 'Yucatan', 'Yunnan', 'Zacatecas', 'Zachodniopomorskie', 'Zaghouan', 'Zagrebacka',
               'Zala', 'Zalec Commune', 'Zanzibar Urban', "Zaporiz'ka Oblast'", 'Zeeland', 'Zhejiang', 'Zilina',
               'Ziri Commune', 'Zlinsky kraj', 'Zrece Commune', 'Zug', 'Zuid-Holland', 'Zurich']

    # Extracting specific columns and removing duplicates
    df_strt = df[['org_uuid', 'org_country_code', 'org_region',  'category_list', 'category_groups_list', 'description']]
    df_strt.drop_duplicates(inplace=True)

    # One-hot encoding the category_list column
    df_encoded_category_list = df_strt['category_list'].str.get_dummies(sep=',')
    df_category_list = pd.concat([df_strt['org_uuid'], df_encoded_category_list], axis=1)

    # One-hot encoding the category_groups_list column
    df_encoded_category_groups = df_strt['category_groups_list'].str.get_dummies(sep=',')
    df_category_groups_list = pd.concat([df_strt['org_uuid'], df_encoded_category_groups], axis=1)

    df_strt_country = df_strt[['org_uuid', 'org_country_code']].pivot_table(index='org_uuid',
                                                                            columns='org_country_code', aggfunc=len,
                                                                            fill_value=0)
    df_strt_region = df_strt[['org_uuid', 'org_region']].pivot_table(index='org_uuid', columns='org_region',
                                                                     aggfunc=len, fill_value=0)

    # Merging the encoded data (this step might need further refinement based on later steps in the notebook)
    df_preprocessed = pd.merge(df_category_list, df_category_groups_list, on='org_uuid', how='left')
    df_preprocessed = pd.merge(df_preprocessed, df_strt_country, on='org_uuid', how='left')
    df_preprocessed = pd.merge(df_preprocessed, df_strt_region, on='org_uuid', how='left')


    df_preprocessed.columns = [col.strip() for col in df_preprocessed.columns]

    df_final = pd.DataFrame(0, index=df_preprocessed.index, columns=columns)

    common_cols = df_preprocessed.columns.intersection(columns)
    df_final[common_cols] = df_preprocessed[common_cols]

    # print(df_final)


    embeddings = model.encode(df_strt['description'].tolist(), convert_to_tensor=True)
    df_strt['embeddings'] = embeddings.tolist()
    embeddings_df = pd.DataFrame(embeddings.cpu())
    df_org_des = pd.concat([df_strt.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)
    df_org_des.drop(columns=['org_country_code', 'org_region', 'category_list', 'category_groups_list'], inplace=True, axis=1)
    df_final = pd.merge(df_org_des, df_final, on='org_uuid', how='left')
    # print(list(df_final.columns))
    print(df_final)

    return df_final


dic_1 = {'org_uuid':['Sisense'], 'org_country_code':['USA'], 'org_region':['Washington'],  'category_list':['Analytics,Big Data,Business Intelligence,Data Visualization,Enterprise Software,Information Technology'], 'category_groups_list':['Data and Analytics,Design,Information Technology,Software'], 'description':['Sisense empowers the builders of analytic apps with powerful tools to simplify complex data and deliver insights to everyone inside and outside their organizations.  Sisense lets builders collaborate on a single platform, delivered in a hybrid, cloud-native environment with the industry‚Äôs lowest cost of ownership, to create true democratization of data and analytics. Sisense has thousands of clients across the globe, including industry leaders like Tinder, Flexport, Philips, Nasdaq and the Salvation Army. Learn more at www.sisense.com.']}

dic_2 = {'org_uuid':['Kids on 45th'], 'org_country_code':['USA'], 'org_region':['New York'],  'category_groups_list':['Clothing and Apparel,Commerce and Shopping,Community and Lifestyle,Design'], 'category_list':['E-Commerce,Fashion,Lifestyle'], 'description':["building a box of kid's everyday essentials with low prices on high quality nearly-new clothing."]}

dic_3 = {'org_uuid':['RIPTec Ltd.'], 'org_country_code':['GB'], 'org_region':['England'],  'category_list':['Software'], 'category_groups_list':['Software'], 'description':["RIPTec Ltd. is a company which specialises in communications apps for SMEs."]}

dic_4 = {'org_uuid':['Scout'], 'org_country_code':['ISR'], 'org_region':['England'],  'category_list':['Software'], 'category_groups_list':['Software'], 'description':["RIPTec Ltd. is a company which specialises in communications apps for SMEs."]}

dic = {'org_uuid':['WindMIL Therapeutics'], 'org_country_code':['USA'], 'org_region':['New Jersey'],  'category_list':['Biotechnology,Medical Device,Therapeutics'], 'category_groups_list':['Biotechnology,Health Care,Science and Engineering'], 'description':["WindMIL Therapeutics is developing cell therapies for oncology indications."]}

df = pd.DataFrame(dic)
feature_vector = preprocess_startup_data(df)
df = feature_vector
df.drop(columns=['org_uuid', 'description', 'embeddings'], inplace=True, axis=1)
recommended_investors_100, rerank_investors_100, recommend_investors_10, rerank_investors_10 = recommend(df.values, dic)
print('RESULT')
print('---------------------------------------------')
#print(f'Top 100 recommended investors: {recommended_investors_100}')
print('---------------------------------------------')
#print(f'RERANKED Top 100 recommended investors: {rerank_investors_100}')
print('---------------------------------------------')
print(f'Top 10 recommended investors with scores: {recommend_investors_10}')
print('---------------------------------------------')
print(f'RERANKED Top 10 recommended investors with scores: {rerank_investors_10}')


