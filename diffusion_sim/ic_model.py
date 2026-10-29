
import os
import pandas as pd
import random
from collections import defaultdict

def build_social_graph(data_dir):
    """Loads the user-user directed graph into an adjacency list."""
    growth_path = os.path.join(data_dir, 'raw', 'flickr-growth.txt.gz')
    df_growth = pd.read_csv(growth_path, sep=r'\t| ', engine='python', header=None, usecols=[0, 1])
    df_growth.columns = ['follower', 'followed']
    
    # Adjacency list: user -> list of followers (people they can influence)
    graph = defaultdict(list)
    for _, row in df_growth.iterrows():
        graph[row['followed']].append(row['follower'])
        
    return graph

def independent_cascade(graph, seed_nodes, prob=0.05):
    """Simulates the spread of a photo through the social graph."""
    activated = set(seed_nodes)
    newly_activated = list(seed_nodes)
    
    step = 0
    while newly_activated:
        next_activations = []
        
        for node in newly_activated:
            # Attempt to influence each follower
            for neighbor in graph[node]:
                if neighbor not in activated:
                    # Roll probability to activate
                    if random.random() < prob:
                        activated.add(neighbor)
                        next_activations.append(neighbor)
                        
        newly_activated = next_activations
        step += 1
        
    return len(activated), step