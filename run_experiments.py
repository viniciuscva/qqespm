import qqespm_module as qq2
import qqsimple_module as qs2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas
from geoobject import GeoObj
from ilquadtree import ILQuadTree
import random
from collections import defaultdict
import pickle
from lat_lon_distance2 import lat_lon_distance
from func_timeout import func_timeout, FunctionTimedOut
from copy import deepcopy
import gc

def get_timestamp():
    from datetime import datetime
    import time
    dt = datetime.now()
    return time.mktime(dt.timetuple()) + dt.microsecond/1e6

def generate_pattern_from_structure(structure, candidate_keywords, qq_module, qualitative_prob, seed = None):
    # if seed is None:
    #     seed = get_timestamp()
    # random.seed(seed)
    vertices_ids = []
    for e in structure:
        vertices_ids.extend(e)
    vertices_ids = list(set(vertices_ids))
    #print('total vertices:', len(vertices_ids))
    keywords = random.sample(candidate_keywords, len(vertices_ids))
    vertices = [ qq_module.SpatialVertex(vertices_ids[i], keywords[i]) for i in range(len(vertices_ids)) ]
    edges = []
    for i, edge in enumerate(structure):
        lij = random.random()*1000 #choose a number between 0 and 1km
        uij = random.random()*10000 + lij + 1#choose a number between lij and 4km
        sign = random.choice(['<','>','<>','-'])
        relation_type = np.random.choice([None, 'related'], size = 1, p = [1-qualitative_prob, qualitative_prob])[0]
        if relation_type == 'related':
            relation = random.choice(['equals', 'touches', 'covers', 'coveredby', 'overlaps', 'disjoint'])
        else:
            relation = None
        edges.append(qq_module.SpatialEdge(i, vertices[edge[0]], vertices[edge[1]], lij, uij, sign, relation))
    sp = qq_module.SpatialPatternGraph(vertices, edges)
    sp.qualitative_prob = qualitative_prob
    return sp, seed

print('Reading and preparing POIs CSV dataset ...')
pois = pd.read_csv('data/pois_paraiba3.csv', low_memory=False)
total_bbox = [-41.418, -9.254, -32.794, -4.544]  # lat long range
pois = pois[['osm_id', 'geometry', 'name', 'amenity', 'shop', 'tourism']]#,'building','office','government']]

pois['geometry'] = geopandas.GeoSeries.from_wkt(pois['geometry'])
def get_centroid(geom):
    if geom.geom_type == 'Point':
        return geom
    return geom.centroid
pois['centroid'] = pois['geometry'].apply(get_centroid)
pois['lon'] = pois['geometry'].apply(get_centroid).apply(lambda e: e.x) # longitudes
pois['lat'] = pois['geometry'].apply(get_centroid).apply(lambda e: e.y) # latitudes
pois.sort_values(by = 'lon', inplace = True)
pois = pois.sample(frac=1)
#pois.fillna('', inplace = True)
print('POIs dataset size:', pois.shape)

# pois.loc[pois['geometry'].apply(lambda e: e.geom_type) == 'LineString', 'geometry'] = pois.loc[pois['geometry'].apply(lambda e: e.geom_type) == 'LineString', 'geometry'].apply(lambda e: e.centroid)
# print('Replace Lines with Points')

distinct_keywords = pois.amenity.value_counts().index.tolist() + pois.shop.value_counts().index.tolist() + pois.tourism.value_counts().index.tolist() 

total_keywords = (pois.shape[0] - pois['amenity'].isna().sum()) + (pois.shape[0] - pois['shop'].isna().sum()) + (pois.shape[0] - pois['tourism'].isna().sum()) 

amenity_totals = pois.amenity.value_counts()
shop_totals = pois.shop.value_counts()
tourism_totals = pois.tourism.value_counts()


most_frequent_keywords = amenity_totals[amenity_totals>100].index.tolist() + \
    shop_totals[shop_totals>100].index.tolist() + \
    tourism_totals[tourism_totals>100].index.tolist()

print('Total of most frequent keywords:', len(most_frequent_keywords))

print('IL-quadtree construction ...')
objs = GeoObj.get_objects_from_geopandas(pois, keyword_columns = ['amenity', 'shop', 'tourism'])#, 'landuse', 'leisure'])#pois.iloc[0:len(pois)//2])#, keyword_columns = ['amenity', 'shop', 'tourism'])

dataset = {
    '20%': ILQuadTree(total_bbox = total_bbox, max_depth = 3),
    '40%': ILQuadTree(total_bbox = total_bbox, max_depth = 3),
    '60%': ILQuadTree(total_bbox = total_bbox, max_depth = 3),
    '80%': ILQuadTree(total_bbox = total_bbox, max_depth = 3),
    '100%': ILQuadTree(total_bbox = total_bbox, max_depth = 3)
}

for percentage in dataset:
    proportion = float(percentage.replace('%','')) / 100.0
    dataset[percentage].insert_elements_from_list(objs[0: int(proportion*len(objs))+1])
    #dataset[percentage].insert_elements_from_geopandas(pois.iloc[0: int(proportion*pois.shape[0])+1])

print('Generating spatial patterns for queries...')
#gerar padrões otimizados
spatial_patterns = []
pattern_structures = [
    [(0,1)], 
    [(0,1),(1,2)],
    [(0,1),(1,2),(2,0)],
    [(0,1),(1,2),(2,3)],
    [(0,1),(1,2),(1,3)],
    [(0,1),(1,2),(2,3),(3,0)],
    [(0,1),(1,2),(2,3),(3,1)],
    [(0,1),(1,2),(2,3),(3,1),(3,4)],
    [(0,1),(1,2),(2,3),(3,4),(4,0)],
    [(0,1),(1,2),(2,3),(3,4),(4,1)],
    [(0,1),(1,2),(2,3),(3,4),(4,1),(1,5)],
    [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(2,5)],
]

qualitative_probs = [1/2, 1/3]


for qualitative_prob in qualitative_probs:
    for structure in pattern_structures:
        for i in range(5):
            sp, seed = generate_pattern_from_structure(structure, most_frequent_keywords, qq2, qualitative_prob)
            spatial_patterns.append(sp)

with open('spatial_patterns3_4.pkl', 'wb') as file:
    pickle.dump(spatial_patterns, file)

#with open('spatial_patterns3_3.pkl', 'rb') as file:
#    spatial_patterns = pickle.load(file)

print('Total spatial patterns:', len(spatial_patterns))

executions = pd.DataFrame(columns = ['repetition',
                                     'num_vertices',
                                     'num_edges', 
                                     'only_frequent_keywords',
                                     'smallest_lij',
                                     'greatest_uij',
                                     'num_qualitative_relations',
                                     'sp',
                                     'seed',
                                     'dataset_size',
                                     'qualitative_prob',
                                     'algorithm',
                                     'exec_time',
                                     'memory_usage',
                                     'total_solutions'])

print('Starting QQESPM executions ...')
# Experimento de escalabilidade do dataset
average_times_scalability_imp = {}
total_repetitions = 3
for repetition in range(total_repetitions):
    for percentage in dataset:
        print('Percentage:', percentage)
        average_times_scalability_imp[percentage] = []
        for i, pattern in enumerate(spatial_patterns):

            solutions, elapsed_time, memory_usage = qq2.QQESPM(dataset[percentage], deepcopy(pattern), False)
            dataset[percentage].cached_existence_searches = {}
            dataset[percentage].cached_searches = {}
            total_solucoes = len(list(solutions))
            average_times_scalability_imp[percentage].append(elapsed_time)
            gc.collect()

            ### CREATING ROW FOR EXECUTIONS DATAFRAME
            num_vertices = len(pattern.vertices)
            num_arestas = len(pattern.edges)
            somente_palavras_frequentes = True
            menor_lij = min([edge.constraint['lij'] for edge in pattern.edges])
            maior_uij = max([-1] + [edge.constraint['uij'] for edge in pattern.edges if edge.constraint['uij'] < float('inf')])
            total_relacoes_qualitativas = len([edge.constraint['relation'] for edge in pattern.edges if edge.constraint['relation'] is not None])
            seed = None#seeds[i]#seeds_by_pattern[pattern]
            dataset_size = int(percentage[:-1])/100 * len(objs)
            qualitative_prob = pattern.qualitative_prob
            algoritmo = 'QQESPM'
            
            row = [repetition,
                 num_vertices,
                 num_arestas, 
                 somente_palavras_frequentes,
                 menor_lij,
                 maior_uij,
                 total_relacoes_qualitativas,
                 pattern.to_json(),
                 seed,
                 dataset_size,
                 qualitative_prob,
                 algoritmo,
                 elapsed_time,
                 memory_usage,
                 total_solucoes]
    
            executions.loc[len(executions)] = row
            ### END CREATING ROW FOR EXECUTIONS DATAFRAME
            executions.to_csv('executions3_4.csv', mode = 'a', index = False, header = False)
            executions = executions[0:0]
        
for percentage in average_times_scalability_imp:
    average_times_scalability_imp[percentage] = np.mean(average_times_scalability_imp[percentage])
print('Average for QQESPM:', average_times_scalability_imp)

print('Ended QQESPM executions.')

print('Starting QQ-simple executions ...')
# Experimento de escalabilidade do dataset
average_times_scalability_dumb = {}
total_repetitions = 3
for repetition in range(total_repetitions):
    for percentage in dataset:
        print('Percentage:', percentage)
        average_times_scalability_dumb[percentage] = []
        for i, pattern in enumerate(spatial_patterns):
            
            solutions, elapsed_time, memory_usage = qs2.QQ_SIMPLE(dataset[percentage], deepcopy(pattern), False)
            dataset[percentage].cached_existence_searches = {}
            dataset[percentage].cached_searches = {}
            total_solucoes = len(list(solutions))
            average_times_scalability_dumb[percentage].append(elapsed_time)
            gc.collect()
            # try:
            #     return_value = func_timeout(600, qs2.QQ_SIMPLE, args=(dataset[percentage], deepcopy(pattern), False))
            #     solutions, elapsed_time, memory_usage = return_value#qq2.QQESPM(dataset[percentage], pattern, debug = False)
            #     dataset[percentage].cached_existence_searches = {}
            #     dataset[percentage].cached_searches = {}
            #     total_solucoes = len(list(solutions))
            #     average_times_scalability_dumb[percentage].append(elapsed_time)
            # except FunctionTimedOut:
            #     print("Could not complete query within 600 seconds and was terminated. Pattern:", pattern)
            #     total_solucoes, elapsed_time, memory_usage = None, float('inf'), None
            #     dataset[percentage].cached_existence_searches = {}
            #     dataset[percentage].cached_searches = {}
                
            ### CREATING ROW FOR EXECUTIONS DATAFRAME
            num_vertices = len(pattern.vertices)
            num_arestas = len(pattern.edges)
            somente_palavras_frequentes = True
            menor_lij = min([edge.constraint['lij'] for edge in pattern.edges])
            maior_uij = max([-1] + [edge.constraint['uij'] for edge in pattern.edges if edge.constraint['uij'] < float('inf')])
            total_relacoes_qualitativas = len([edge.constraint['relation'] for edge in pattern.edges if edge.constraint['relation'] is not None])
            seed = None#seeds[i]#seeds_by_pattern[pattern]
            dataset_size = int(percentage[:-1])/100 * len(objs)
            qualitative_prob = pattern.qualitative_prob
            algoritmo = 'QQ-simple'
            
            row = [repetition,
                 num_vertices,
                 num_arestas, 
                 somente_palavras_frequentes,
                 menor_lij,
                 maior_uij,
                 total_relacoes_qualitativas,
                 pattern.to_json(),
                 seed,
                 dataset_size,
                 qualitative_prob,
                 algoritmo,
                 elapsed_time,
                 memory_usage,
                 total_solucoes]
    
            executions.loc[len(executions)] = row
            ### END CREATING ROW FOR EXECUTIONS DATAFRAME
            executions.to_csv('executions3_4.csv', mode = 'a', index = False, header = False)
            executions = executions[0:0]

        
for percentage in average_times_scalability_dumb:
    average_times_scalability_dumb[percentage] = np.mean(average_times_scalability_dumb[percentage])
print('Average for QQ-simple:', average_times_scalability_dumb)
print('Ended QQ-simple executions.')
