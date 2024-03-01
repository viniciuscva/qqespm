from ilquadtree import ILQuadTree, get_depth, get_MBR, dmin, dmax, bboxes_intersect, get_nodes, get_nodes_at_level
from geoobject import GeoObj
import json
import multiprocessing
from multiprocessing.pool import ThreadPool
from time import time
import psutil
from functools import partial
from itertools import product as cartesian_product
from lat_lon_distance2 import lat_lon_distance
import itertools
from time import sleep
import json

#import matplotlib.pyplot as plt
#import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout
#import math

global translations
translations = {
 'touches': 'conectado',
 'disjoint': 'desconectado',
 'covers': 'cobre',
 'coveredby': 'coberto por',
 'medical_supply': 'loja de suprimentos médicos',
 'stationery': 'papelaria',
 'hairdresser_supply': 'loja de acessórios para cabeleireiro',
 'watches': 'loja de relógios',
 'gas; water': 'gás; loja de água',
 'stones': 'loja de pedras',
 'general': 'loja geral',
 'fuel': 'combustível',
 'funeral_directors': 'loja de agentes funerários',
 'optician': 'ótica',
 'bed': 'loja de cama',
 'doors': 'loja de portas',
 'plants': 'loja de plantas',
 'electronics': 'loja de eletrônicos',
 'video': 'loja de video',
 'locksmith': 'serralheria',
 'ice_cream': 'sorvete',
 'greengrocer': 'quitanda',
 'paint': 'oficina de pintura',
 'books': 'loja de livros',
 'craft': 'loja de artesanato',
 'swimming_pool': 'loja de piscina',
 'fashion_accessories': 'loja de acessórios de moda',
 'tailor': 'alfaiataria',
 'anime': 'loja de anime',
 'show': 'show shop',
 'copyshop': 'copiadora',
 'bicycle': 'Loja de bicicletas',
 'newsagent': 'banca de jornal',
 'tiles': 'loja de azulejos',
 'hairdresser': 'salão de cabeleireiro',
 'wine': 'loja de vinhos',
 'fire_extinguisher': 'loja de extintores',
 'car_repair': 'loja de reparação de carros',
 'Andaimes_e_Escoramentos': 'Loja Andaimes e Escoramentos',
 'pipes': 'loja de cachimbos',
 'yes': 'sim',
 'loja_de_calçados': 'loja de calçados',
 'appliance': 'loja de eletrodomésticos',
 'chocolate': 'Loja de chocolates',
 'religion': 'loja de religião',
 'baby_goods': 'loja de artigos para bebês',
 'water_sports': 'loja de esportes aquáticos',
 'bookmaker': 'loja de apostas',
 'musical_instrument': 'loja de instrumentos musicais',
 'frame': 'loja de molduras',
 'nutrition_supplements': 'loja de suplementos nutricionais',
 'sports': 'Loja de esportes',
 'motorcycle_repair': 'oficina de motos',
 'car': 'loja de carros',
 'pet': 'loja de animais',
 'tyres': 'loja de pneus',
 'no': 'nenhuma loja',
 'art': 'loja de artes',
 'sewing': 'loja de costura',
 'farm': 'loja rural',
 'health_food': 'comida saudável',
 'butcher': 'açougue',
 'accessories': 'loja de acessórios',
 'department_store': 'loja de departamento',
 'fabric': 'loja de tecidos',
 'loja_de_som_para_carros': 'loja de som para carros',
 'kitchen': 'loja de cozinha',
 'jewelry': 'Joalheria',
 'kiosk': 'loja de quiosque',
 'seafood': 'loja de frutos do mar',
 'chemist': 'Farmácia',
 'electrical': 'loja Elétrica',
 'toys': 'loja de brinquedos',
 'cheese': 'queijaria',
 'glaziery': 'vidraçaria',
 'deli': 'loja de delicatessen',
 'cosmetics': 'loja de cosméticos',
 'erotic': 'loja erótica',
 'Administrative_Services': 'loja de serviços administrativos',
 'spices': 'loja de especiarias',
 'outdoor': 'loja ao ar livre',
 'bag': 'loja de bolsas',
 'eletronics': 'loja de eletrônicos',
 'mobile_phone': 'loja de celulares',
 'vacant': 'loja vazia',
 'photo': 'loja de fotos',
 'perfumery': 'loja de perfumaria',
 'coffee': 'cafeteria',
 'garden_centre': 'loja de jardinagem',
 'hunting': 'loja de caça',
 'wholesale': 'loja de atacado',
 'loja_do_colchão': 'loja do colchão shop',
 'tattoo': 'loja de tatuagem',
 'candles': 'loja de velas',
 'country_store': 'loja de loja do país',
 'household_linen': 'loja de roupa de casa',
 'hardware': 'loja de ferragens',
 'fishing': 'loja de pesca',
 'party': 'loja de festas',
 'ticket': 'loja de bilhetes',
 'games': 'loja de jogos',
 'alcohol': 'loja de bebidas alcoólicas',
 'construction': 'loja de construção',
 'tea': 'loja de chá',
 'lottery': 'lotérica',
 'variety_store': 'loja de variedades',
 'travel_agency': 'loja de agência de viagens',
 'place': 'loja de lugar',
 'interior_decoration': 'loja de decoração de interiores',
 'dry_cleaning': 'loja de limpeza a seco',
 'estate_agent': 'loja de agente imobiliário',
 'convenience': 'loja de conveniência',
 'bakery': 'padaria',
 'hearing_aids': 'loja de aparelhos auditivos',
 'gas': 'loja de gás',
 'shoes': 'loja de sapatos',
 'shoe_repair': 'sapataria',
 'pet_grooming': 'pet shop',
 'carpet': 'loja de tapetes',
 'music': 'loja de música',
 'money_lender': 'loja de agiotas',
 'massage': 'loja de massagens',
 'car_parts': 'loja de autopeças',
 'pastry': 'pastelaria',
 'agrarian': 'loja agrária',
 'herbalist': 'loja de ervas',
 'gift': 'loja de presentes',
 'radiotechnics': 'loja de radiotecnia',
 'supermarket': 'loja de supermercado',
 'water': 'loja de água',
 'repair': 'oficina',
 'storage_rental': 'loja de aluguel de armazenamento',
 'laundry': 'lavanderia',
 'houseware': 'loja de utensílios domésticos',
 'telecommunication': 'loja de telecomunicações',
 'tobacco': 'tabacaria',
 'motorcycle': 'loja de motos',
 'doityourself': 'loja faça-você-mesmo',
 'lighting': 'loja de iluminação',
 'clothes': 'loja de roupas',
 'rental': 'locadora',
 'Loja_de_Iluminação': 'Loja de Iluminação',
 'pasta': 'loja de massas',
 'beauty': 'salao de beleza',
 'leather': 'loja de couro',
 'mall': 'loja de shopping',
 'trade': 'loja comercial',
 'florist': 'floricultura',
 'lan_house_personalizados_casa_da_xerox_recife': 'lan house personalizados casa da xerox loja recife',
 'fornecedor_de_revestimento': 'loja fornecedora de revestimento',
 'computer': 'loja de informática',
 'bicycle_repair': 'oficina de bicicletas',
 'confectionery': 'confeitaria',
 'window_blind': 'loja de persianas',
 'confeitaria': 'loja de confeitaria',
 'flooring': 'loja de pisos',
 'beverages': 'loja de bebidas',
 'construtora': 'loja da construtora',
 'yes;clothes': 'sim; loja de roupas',
 'energy': 'loja de energia',
 'furniture': 'loja de móveis',
 'video_games': 'loja de videogames',
 'animal_breeding': 'criação de animais',
 'pumping_station': 'estação de bombeamento',
 'vending_machine': 'Maquina de vendas',
 'food_court': 'Praça de alimentação',
 'gambling': 'jogatina',
 'shower': 'banho',
 'harbourmaster': 'capitão do porto',
 'community_centre': 'Centro Comunitário',
 'driving_school': 'escola de condução',
 'steak house': 'Churrascaria',
 'roro_terminal': 'terminal roro',
 'nightclub': 'Boate',
 'research_institute': 'Instituto de Pesquisa',
 'bus_station': 'Rodoviária',
 'fire_station': 'Corpo de Bombeiros',
 'public_bath': 'banheiro público',
 'monastery': 'mosteiro',
 'bureau_de_change': 'casa de câmbio',
 'drinking_water': 'água potável',
 'social_centre': 'centro social',
 'telephone': 'Telefone',
 'grave_yard': 'cemitério',
 'bicycle_repair_station': 'estação de reparação de bicicletas',
 'weighbridge': 'báscula',
 'scrapyard': 'ferro-velho',
 'music_school': 'Escola de música',
 'bar': 'bar',
 'container': 'recipiente',
 'exhibition_centre': 'Centro de exibição',
 'mortuary': 'necrotério',
 'waste': 'desperdício',
 'atm': 'atm',
 'slaughterhouse': 'matadouro',
 'language_school': 'escola de idiomas',
 'water_point': 'ponto de água',
 'boat_rental': 'aluguel de barco',
 'television': 'televisão',
 'bicycle_parking': 'estacionamento de bicicletas',
 'fountain': 'fonte',
 'bbq': 'churrasco',
 'shelter': 'abrigo',
 'place_of_worship': 'local de culto',
 'theatre': 'teatro',
 'watering_place': 'bebedouro',
 'library': 'biblioteca',
 'toilets': 'banheiros',
 'taxi': 'Táxi',
 'building': 'prédio',
 'post_depot': 'depósito postal',
 'waste_basket': 'cesto de lixo',
 'biergarten': 'biergarten',
 'clock': 'relógio',
 'hall': 'salão',
 'register_office': 'cartório',
 'bench': 'banco',
 'townhall': 'Prefeitura',
 'recycling': 'reciclando',
 'prison': 'prisão',
 'grocery': 'mercado',
 'track': 'acompanhar',
 'courthouse': 'tribunal',
 'marketplace': 'Mercado',
 'bank': 'banco',
 'tec_common': 'técnico comum',
 'childcare': 'puericultura',
 'hospital': 'hospital',
 'police': 'polícia',
 'cinema': 'cinema',
 'car_rental': 'aluguel de carros',
 'animal_shelter': 'abrigo de animais',
 'arts_centre': 'centro de artes',
 'doctors': 'médicos',
 'post_box': 'caixa postal',
 'fast_food;restaurant': 'restaurante fast food',
 'compressed_air': 'ar comprimido',
 'nursing_home': 'casa de repouso',
 'studio': 'estúdio',
 'prep_school': 'escola Preparatória',
 'parking': 'estacionamento',
 'restaurant': 'restaurante',
 'love_hotel': 'hotel do amor',
 'animal_boarding': 'embarque de animais',
 'conference_centre': 'Centro de conferências',
 'payment_centre': 'centro de pagamento',
 'lavoir': 'lavoir',
 'internet_cafe': 'cibercafé',
 'dentist': 'dentista',
 'fixme': 'conserte-me',
 'sidewalk': 'calçada',
 'dressing_room': 'provador',
 'university': 'universidade',
 'clinic': 'clínica',
 'veterinary': 'veterinário',
 'waste_disposal': 'depósito de lixo',
 'bicycle_rental': 'aluguer de bicicletas',
 'waste_transfer_station': 'estação de transferência de resíduos',
 'crematorium': 'crematório',
 'public_building': 'edifício público',
 'charging_station': 'estação para carregar',
 'hunting_stand': 'estande de caça',
 'office': 'escritório',
 'events_venue': 'local de eventos',
 'stripclub': 'clube de strip-tease',
 'kindergarten': 'Jardim da infância',
 'toy_library': 'brinquedoteca',
 'payment_terminal': 'terminal de pagamento',
 'pub': 'bar',
 'parking_space': 'espaço de estacionamento',
 'events_centre': 'centro de eventos',
 'pharmacy': 'farmacia',
 'fast_food': 'comida rápida',
 'college': 'faculdade',
 'social_facility': 'instalação social',
 'cafe': 'cafeteria',
 'dojo': 'dojo',
 'industrial': 'industrial',
 'parking_entrance': 'entrada do estacionamento',
 'ferry_terminal': 'terminal da balsa',
 'post_office': 'correios',
 'festival_grounds': 'recinto do festival',
 'car_wash': 'lava-jato',
 'vehicle_inspection': 'inspeção veicular',
 'school': 'escola',
 'motorcycle_parking': 'estacionamento de moto',
 'ticket_validator': 'validador de bilhetes',
 'car_sharing': 'compartilhamento de carros',
 'casino': 'cassino',
 'guest_house': 'casa de hóspedes',
 'picnic_site': 'local de piquenique',
 'viewpoint': 'ponto de vista',
 'artwork': 'obra de arte',
 'museum': 'museu',
 'gallery': 'galeria',
 'hostel': 'Hostel',
 'resort': 'recorrer',
 'zoo': 'jardim zoológico',
 'chalet': 'chalé',
 'atr': 'atr',
 'attraction': 'atração',
 'camp_pitch': 'campo de acampamento',
 'camp_site': 'acampamento',
 'apartment': 'apartamento',
 'aquarium': 'aquário',
 'information': 'Informação',
 'theme_park': 'Parque temático',
 'gastronomy': 'gastronomia',
 'caravan_site': 'local de caravana',
 'wilderness_hut': 'cabana selvagem',
 'hotel': 'hotel',
 'alpine_hut': 'cabana alpina',
 'motel': 'motel',
 'trail_riding_station': 'estação de trilha',
 '': ''}


def translate(word):
    global translations
    return translations.get(word, word)


class SpatialVertex:
    def __init__(self, id, keyword):
        self.id = id
        self.keyword = keyword
    def __str__(self):
        return str(self.id) + '(' + str(self.keyword) + ')'
    def __hash__(self):
        return hash(self.__str__())
    def __repr__(self):
        return str(self.id) + '(' + str(self.keyword) + ')'

    def __eq__(self, another_vertex):
        return self.id == another_vertex.id and self.keyword == another_vertex.keyword

    @staticmethod
    def from_id(id, vertices):
        for v in vertices:
            if v.id == id:
                return v
        return None
    
class SpatialMultiVertex:
    def __init__(self, id, keywords):
        self.id = id
        self.keywords = keywords
        self.vertices = [SpatialVertex(str(id)+'-'+str(i), keyword) for i, keyword in enumerate(keywords)]
    def __str__(self):
        return str(self.id) + '(' + str(self.keywords) + ')'
    
class SpatialEdge:
    def __init__(self, id, vi, vj, lij = 0, uij = float('inf'), sign = '-', relation = None):
        # constraint should be a dict like {'lij':0, 'uij':1000, 'sign':'>', 'relation': disjoint}
        # 'sign' is always of of the four {'>', '<', '<>', '-'}
        # 'relation' should be a string, specifying the type of topological relation 
        # possible relations: equals, touches, overlaps, covers, coveredby, disjoint
        self.id = id
        self.vi = vi
        self.vj = vj
        if relation is not None and relation != 'disjoint':
            lij = 0
            uij = float('inf')
            sign = '-'
        self.constraint = {'lij': lij, 'uij': uij, 'sign': sign, 'relation': relation}
        self.constraint['is_exclusive'] = False if self.constraint['sign']=='-' else True
    def __str__(self):
        return str(self.id) + ': ' + str(self.vi) + ' ' + self.constraint['sign'] + ' ' + str(self.vj) + ' (' + str(self.constraint) + ')'

    def __eq__(self, another):
        return self.id == another.id and self.vi == another.vi and self.vj == another.vj and self.constraint['lij'] == another.constraint['lij'] and \
            self.constraint['uij'] == another.constraint['uij'] and self.constraint['sign'] == another.constraint['sign'] and \
            self.constraint['relation'] == another.constraint['relation']

    def __hash__(self):
        return hash(self.__str__())

    def get_contraint_label(self):
        label = ""
        relation = self.constraint['relation']
        if relation is None or relation == 'disjoint':
            if self.constraint['lij'] > 0:
                label += f"minimum distance: {round(self.constraint['lij'],3)}\n"
            label += f"maximum distance: {round(self.constraint['uij'],3)}\n"
        if relation is not None:
            label += f"{translate(self.constraint['relation'])}\n"
        return label[:-1]
    
    @staticmethod
    def get_edge_by_id(edges, id):
        for edge in edges:
            if edge.id == id:
                return edge
    
    
def find_edge(vertex_i, vertex_j, edges):
    for edge in edges:
        if edge.vi == vertex_i and edge.vj == vertex_j:
            return edge
    return None
    
class SpatialPatternMultiGraph:
    def __init__(self, multi_vertices, edges):
        # vertices should be a list of SpatialVertex objects 
        # edges should be a list of SpatialEdge objects
        self.pattern_type = 'Multi_keyword_vertices_graph'
        self.multi_vertices = multi_vertices
        self.edges = edges
        
        self.spatial_patterns = []
        keywords_of_vertices = [multi_vertex.keywords for multi_vertex in multi_vertices]
        for keywords_choice in cartesian_product(*keywords_of_vertices):
            simples_vertices = [SpatialVertex(i, wi) for i, wi in enumerate(keywords_choice)]
            simple_edges = []
            for i, multi_vertex_i in enumerate(multi_vertices):
                for j, multi_vertex_j in enumerate(multi_vertices):
                    edge_found = find_edge(multi_vertex_i, multi_vertex_j, edges)
                    if edge_found is not None:
                        lij, uij = edge_found.constraint['lij'], edge_found.constraint['uij']
                        sign, relation = edge_found.constraint['sign'], edge_found.constraint['relation']
                        simple_edges.append(SpatialEdge(str(i)+'-'+str(j), simples_vertices[i], simples_vertices[j], lij, uij, sign, relation))
        self.spatial_patterns.append(SpatialPatternGraph(simples_vertices, simple_edges))
        # self.matrix = [[0]*len(vertices) for _ in range(len(vertices))] # just initializing
        # for edge in edges:
        #     vi, vj = edge.vi, edge.vj
        #     i, j = vertices.index(vi), vertices.index(vj)
        #     self.matrix[i][j] = edge
    def __str__(self):
        descr = ""
        for edge in self.edges:
            descr += edge.__str__() + '\n'
        return descr
    
class SpatialPatternGraph:
    def __init__(self, *args):
        # vertices should be a list of SpatialVertex objects 
        # edges should be a list of SpatialEdge objects
        if len(args) == 2:
            vertices, edges = args
            self.pattern_type = 'simple_graph'
            self.vertices = vertices
            self.edges = edges
        elif len(args) == 1:
            self.from_json(args[0])

    def from_json(self, json_str):
        sp_dict = json.loads(json_str)
        vertices = []
        for vertex_id, vertex_keyword in sp_dict['vertices'].items():
            vertices.append(SpatialVertex(int(vertex_id), vertex_keyword))
        edges = []
        for edge_id, edge_data in sp_dict['edges'].items():
            edges.append(SpatialEdge(edge_id, 
                                     SpatialVertex.from_id(edge_data['vi'], vertices),
                                     SpatialVertex.from_id(edge_data['vj'], vertices),
                                     float(edge_data['lij']),
                                     float(edge_data['uij']),
                                     edge_data['sign'],
                                     edge_data['relation']
                        ))
        self.pattern_type = 'simple_graph'
        self.vertices = vertices
        self.edges = edges
        
    def __str__(self):
        descr = ""
        for edge in self.edges:
            descr += edge.__str__() + '\n'
        return descr

    def __eq__(self, another):
        ordered_vertices = sorted(self.vertices, key = lambda e: e.id)
        ordered_edges = sorted(self.edges, key = lambda e: e.id)
        ordered_vertices_another = sorted(another.vertices, key = lambda e: e.id)
        ordered_edges_another = sorted(another.edges, key = lambda e: e.id)
        return len(ordered_vertices) == len(ordered_vertices_another) and \
                len(ordered_edges) == len(ordered_edges_another) and \
                all([(ordered_vertices[i] == ordered_vertices_another[i]) for i in range(len(ordered_vertices))]) and \
                all([(ordered_edges[i] == ordered_edges_another[i]) for i in range(len(ordered_edges))])
        
    def __hash__(self):
        return hash(self.to_json())

    def __lt__(self, another):
        return self.__hash__() < another.__hash__()

    def to_json(self, indent = None):
        ordered_vertices = sorted(self.vertices, key = lambda e: e.id)
        ordered_edges = sorted(self.edges, key = lambda e: e.id)
        sp_dict = {
            "vertices": {
                v.id: v.keyword for v in ordered_vertices
            },
            "edges": {
                e.id: {
                    "vi": e.vi.id,
                    "vj": e.vj.id,
                    "lij": e.constraint['lij'],
                    "uij": e.constraint['uij'],
                    "sign": e.constraint['sign'],
                    "relation": e.constraint['relation'],
                } for e in ordered_edges
            },
        }
        return json.dumps(sp_dict, indent=indent, ensure_ascii=False)#.encode('utf8').decode()

    # def draw(self):
    #     G = nx.Graph()
    #     plt.figure(figsize=(7,7))
    #     nx_edges = {(e.vi.keyword, e.vj.keyword): e.get_contraint_label() for e in self.edges}
    #     G.add_edges_from(list(nx_edges.keys()))
    #     pos = graphviz_layout(G)#, k=0.5)#, iterations=50)
    #     nx.draw(
    #         G, pos, edge_color='black', width=1, linewidths=1, 
    #         node_size=3000, node_color='white', alpha=1,
    #         labels={node: translate(node) for node in G.nodes()}
    #     )
    #     nx.draw_networkx_edge_labels(
    #         G, pos,
    #         edge_labels={e: nx_edges[e] for e in nx_edges},
    #         font_color='red'
    #     )
    #     l,r = plt.xlim()
    #     #print(l,r)
    #     plt.xlim(l-2,r+2)
    #     plt.show()
    
    
def is_e_match(ilq: ILQuadTree, os, ot, edge: SpatialEdge):
    # this verification bellow is not necessary if the node matches are computed correctly
    # if not(edge.vi.keyword in os.keywords()) or not(edge.vj.keyword in ot.keywords()):
    #     return False
    
    lij, uij = edge.constraint['lij'], edge.constraint['uij']
    distance = os.distance(ot)
    if not (lij <= distance <= uij):
        return False
    
    # if edge.constraint['relation'] is not None and edge.constraint['relation'] != os.relation(ot):
    #     return False
    
    vi, vj = edge.vi, edge.vj
    

    if edge.constraint['sign']=='>': #vi excludes vj
        # there should not be any object with vj's keyword nearer than lij from os
        if ilq.search_circle_existence((vj.keyword,), os.centroid(), lij):
            return False
    elif edge.constraint['sign']=='<': #vj excludes vi
        # there should not be any object with vi's keyword nearer than lij from ot
        if ilq.search_circle_existence((vi.keyword,), ot.centroid(), lij):
            return False
    elif edge.constraint['sign']=='<>': #vj mutual exclusion with vi
        # there should not be any object with vi's keyword nearer than lij from ot
        # and also, there should not be any object with vj's keyword nearer than lij from os
        if ilq.search_circle_existence((vj.keyword,), os.centroid(), lij):
            return False
        if ilq.search_circle_existence((vi.keyword,), ot.centroid(), lij):
            return False
    return True
            
    
def is_n_match(ILQi, ILQj, node_i, node_j, edge: SpatialEdge, ilq):
    # node_i e node_j are of type pyqtree._QuadNode
    bi = get_MBR(node_i)
    bj = get_MBR(node_j)
    lij = edge.constraint['lij']
    uij = edge.constraint['uij']

    #d_min = dmin(bi,bj)
    # if d_min is None:
    #     print('dmin is none. bi,bj:', bi, bj)
    #     sleep(1)
    #     return "Error, dmin is none!"
    if not (dmin(bi,bj) <= uij and dmax(bi,bj) >= lij):
        return False
    
    if edge.constraint['sign'] == '-':
        return True
        
    elif edge.constraint['sign'] == '>':
        # we will do a radius search centered on the center point of node_i, and with radius max(0, lij-r(node_i))
        # r(node_i) represents the distance between the center of node_i and one of its extreme vertices.
        xci,yci = node_i.center
        xv,yv,_,_ = bi
        r_node_i = lat_lon_distance(yci, xci, yv, xv)
        radius = max(0, lij - r_node_i)
        result = ilq.search_circle_existence((edge.vj.keyword,), (xci,yci), radius)
        if result:
            return False
        return True
        
    elif edge.constraint['sign'] == '<':
        # we will do a radius search centered on the center point of node_j, and with radius max(0, lij-r(node_j))
        # r(node_j) represents the distance between the center of node_j and one of its extreme vertices.
        xcj,ycj = node_j.center
        xv,yv,_,_ = bj
        r_node_j = lat_lon_distance(ycj, xcj, yv, xv)
        radius = max(0, lij - r_node_j)
        result = ilq.search_circle_existence((edge.vi.keyword,), (xcj,ycj), radius)        
        if result:
            return False
        return True
    else:
        xci,yci = node_i.center
        xv,yv,_,_ = bi
        r_node_i = lat_lon_distance(yci, xci, yv, xv)
        radius = max(0, lij - r_node_i)
        result1 = ilq.search_circle_existence((edge.vj.keyword,), (xci,yci), radius)
        if result1:
            return False
        
        xcj,ycj = node_j.center
        xv,yv,_,_ = bj
        r_node_j = lat_lon_distance(ycj, xcj, yv, xv)
        radius = max(0, lij - r_node_j)
        result2 = ilq.search_circle_existence((edge.vi.keyword,), (xcj,ycj), radius)
        if result2:
            return False
        return True


def find_sub_n_matches(n_match, candidate_nodes_vi, candidate_nodes_vj, ILQi, ILQj, edge, ilq):
    n_matches_l = []
    node_i, node_j = n_match
    children_i = get_nodes_at_level(node_i, 1)
    children_j = get_nodes_at_level(node_j, 1)
    #print('teste0', len(children_i), len(children_j))
    if candidate_nodes_vi != set():
        # the intersection of children_i and candidate_nodes_vi
        children_i = filter(set(children_i).__contains__, candidate_nodes_vi)
        #print('teste1')
    if candidate_nodes_vj != set():
        children_j = filter(set(children_j).__contains__, candidate_nodes_vj)
        #print('teste2')
    for ci in children_i:
        for cj in children_j:
            if is_n_match(ILQi, ILQj, ci, cj, edge, ilq):
                n_matches_l.append((ci,cj))
                #print(f'Teste/ edge {edge}\n n_matches_l: {n_matches_l}')
    return n_matches_l


def compute_n_matches_at_level_parallel(ilq: ILQuadTree, edge: SpatialEdge, level: int, previous_n_matches: list, candidate_nodes_vi = set(), candidate_nodes_vj = set(), debug = False):
    ilq.clean_cached_searches()
    #print('Before computation of qq-n-matches level - len(cache_circle_searches):', len(ilq.cached_searches))
    pool_obj = pool_obj = ThreadPool(int(multiprocessing.cpu_count()-1))
    wi, wj = edge.vi.keyword, edge.vj.keyword
    ILQi = ilq.quadtrees[wi]
    ILQj = ilq.quadtrees[wj]
    n_matches_l = []
    find_sub_n_matches_partial = partial(find_sub_n_matches, candidate_nodes_vi = candidate_nodes_vi, candidate_nodes_vj = candidate_nodes_vj, ILQi = ILQi, ILQj = ILQj, edge = edge, ilq = ilq )

    # option 1: parallel
    #print('Before map. len(candidate_nodes_vi), len(candidate_nodes_vj), len(previous_n_matches):', len(candidate_nodes_vi), len(candidate_nodes_vj), len(previous_n_matches))
    results = pool_obj.map(find_sub_n_matches_partial, previous_n_matches)
    #results, caches = list(zip(*results_with_cache_searches))
    #cache_circle_searches = caches[-1]
    n_matches_l = list(itertools.chain(*results))
    #print('After computation of qq-n-matches level - len(cached_searches):', len(ilq.cached_searches))
    # option 2: sequential
    #print('chose sequential for qq-n-matches level')
    #for n_mt in previous_n_matches:
    #    find_sub_n_matches_partial(n_mt)

    
    #n_matches_l = list(n_matches_l)
    #print(f"Total n-matches at level {level} for edge {str(edge)}: {len(n_matches_l)}\n len(previous_n_matches): {len(previous_n_matches)}")
    pool_obj.close()
    return n_matches_l        

# def compute_n_matches_at_level_with_codes(ilq: ILQuadTree, edge: SpatialEdge, level: int, previous_n_matches: list, previous_n_matches_code_pairs, candidate_nodes_vi = set(), candidate_nodes_vj = set()):
#     wi, wj = edge.vi.keyword, edge.vj.keyword
#     ILQi, ILQj = ilq.quadtrees[wi], ilq.quadtrees[wj]
#     n_matches_l = []
#     n_matches_code_pairs_l = []
#     for i,(node_i, node_j) in enumerate(previous_n_matches):
#         children_i = get_nodes_at_level(node_i, 1)
#         if len(children_i) == 1: # node doesn't have children so it's children are set to be a list with only itself as element
#             children_i[0] = ('', children_i[0]) # a pair: subtree code, subtree itself
#         else: # so it's 4
#             children_i[0] = ('00', children_i[0])
#             children_i[1] = ('10', children_i[1])
#             children_i[2] = ('01', children_i[2])
#             children_i[3] = ('11', children_i[3])
#         children_j = get_nodes_at_level(node_j, 1)
#         if len(children_j) == 1: # node doesn't have children so it's children are set to be a list with only itself as element
#             children_j[0] = ('', children_j[0]) # a pair: subtree code, subtree itself
#         else: # so it's 4
#             children_j[0] = ('00', children_j[0])
#             children_j[1] = ('10', children_j[1])
#             children_j[2] = ('01', children_j[2])
#             children_j[3] = ('11', children_j[3])
#         if candidate_nodes_vi != set():
#             children_i = [c for c in children_i if c[1] in candidate_nodes_vi]
#         if candidate_nodes_vj != set():
#             children_j = [c for c in children_j if c[1] in candidate_nodes_vj]
#         for ci in children_i:
#             for cj in children_j:
#                 if is_n_match(ILQi, ILQj, ci[1], cj[1], edge):
#                     n_matches_l.append((ci[1],cj[1]))
#                     n_matches_code_pairs_l.append((previous_n_matches_code_pairs[i][0]+ci[0],
#                                         previous_n_matches_code_pairs[i][1]+cj[0]))
#     #n_matches_l = list(n_matches_l)
#     #print(f'Total n_matches at level {level} for edge {(edge.vi.keyword, edge.vj.keyword)}: {len(n_matches_l)}')
#     #print(n_matches_code_pairs_l)
#     return n_matches_l, n_matches_code_pairs_l
                                                                       

def compute_n_matches_for_all_edges(ilq: ILQuadTree, sp: SpatialPatternGraph, debug = False):
    #t0 = time()
    edges = sp.edges
    vertices = sp.vertices
    keywords = [v.keyword for v in vertices]
    depth = max([get_depth(ilq.quadtrees[keyword]) for keyword in keywords])
    #n_matches_by_level = {}
    #print('depth:', depth)
    # we need to reorder edges array to an optimal ordering to minimize computation efforts
    # 1) it partitions edges into two groups, where the first group
    # contains exclusive edges and the second group contains mutually
    # inclusive edges; 2) for each group, it ranks edges in an ascending
    # order of numbers of their n-matches in the previous level; and 3) by
    # concatenating edges in these two groups, it obtains the order of edges
    # for computing n-matches.
    exclusive_edges = [edge for edge in edges if edge.constraint['is_exclusive']]
    inclusive_edges = [edge for edge in edges if not edge.constraint['is_exclusive']]
    n_matches_exclusive = dict()
    previous_n_matches_exclusive = dict()
    n_matches_inclusive = dict()
    previous_n_matches_inclusive = dict()
    for ee in exclusive_edges:
        #print('exclusive edge:', ee)
        wi, wj = ee.vi.keyword, ee.vj.keyword
        previous_n_matches_exclusive[ee] = [(ilq.quadtrees[wi], ilq.quadtrees[wj])]
    for ie in inclusive_edges:
        #print('inclusive edge:', ie)
        wi, wj = ie.vi.keyword, ie.vj.keyword
        previous_n_matches_inclusive[ie] = [(ilq.quadtrees[wi], ilq.quadtrees[wj])]
    candidate_nodes = dict()
    #if len(edges) == 1:
    #    f_compute_n_matches_at_level = compute_n_matches_at_level
    #else:
    f_compute_n_matches_at_level = compute_n_matches_at_level_parallel
    for level in range(1, max(2,depth+1)):
        #print('level =', level)
        #print('Computing n-matches at level', level)
        for vertex in vertices:
            candidate_nodes[vertex] = set() # it is the set of nodes that are candidates to this vertex in this level
            
        for ee in exclusive_edges:
            #print('level, edge:', level, ee)
            n_matches_exclusive[ee] = f_compute_n_matches_at_level(ilq, ee, level, previous_n_matches_exclusive[ee], candidate_nodes[ee.vi], candidate_nodes[ee.vj], debug = debug)
            #print(f'Total qq-n-matches for current edge at level {level}: {len(n_matches_exclusive[ee])}')
            if len(n_matches_exclusive[ee]) == 0:
                return 
            previous_n_matches_exclusive[ee] = n_matches_exclusive[ee]
            new_candidates_i, new_candidates_j = zip(*n_matches_exclusive[ee])
            if candidate_nodes[ee.vi]==set(): candidate_nodes[ee.vi] = set(new_candidates_i)
            else: candidate_nodes[ee.vi] = candidate_nodes[ee.vi].intersection(set(new_candidates_i))
            if candidate_nodes[ee.vj]==set(): candidate_nodes[ee.vj] = set(new_candidates_j)
            else: candidate_nodes[ee.vj] = candidate_nodes[ee.vj].intersection(set(new_candidates_j))
            
        for ie in inclusive_edges:
            #print('level, edge:', level, ie)
            n_matches_inclusive[ie] = f_compute_n_matches_at_level(ilq, ie, level, previous_n_matches_inclusive[ie], candidate_nodes[ie.vi], candidate_nodes[ie.vj], debug = debug)
            if len(n_matches_inclusive[ie]) == 0:
                return 
            previous_n_matches_inclusive[ie] = n_matches_inclusive[ie]
            new_candidates_i, new_candidates_j = zip(*n_matches_inclusive[ie])
            if candidate_nodes[ie.vi]==set(): candidate_nodes[ie.vi] = set(new_candidates_i)
            else: candidate_nodes[ie.vi] = candidate_nodes[ie.vi].intersection(set(new_candidates_i))
            if candidate_nodes[ie.vj]==set(): candidate_nodes[ie.vj] = set(new_candidates_j)
            else: candidate_nodes[ie.vj] = candidate_nodes[ie.vj].intersection(set(new_candidates_j))
            
        # sort list  exclusive_edges according to len(n_matches_exclusive[ee])
        # also sort the list inclusive_edges according to len(n_matches_inclusive[ie])
        #n_matches_by_level[level] = {**n_matches_exclusive.copy(), **n_matches_inclusive.copy()}
        exclusive_edges.sort(key = lambda ee: len(n_matches_exclusive[ee]))
        inclusive_edges.sort(key = lambda ie: len(n_matches_inclusive[ie]))
        # print(level, 'edges with n-matches (exc,inc):', len(n_matches_exclusive), len(n_matches_inclusive))
        # print(level, 'total edges:', len(edges))
        # if len(n_matches_exclusive) + len(n_matches_inclusive) != len(edges):
        #     print(edges)
        #     print('---')
        #     print(n_matches_exclusive)
        #     print('---')
        #     print(n_matches_inclusive)
        
    #print('edges with n-matches (exc,inc):', len(n_matches_exclusive), len(n_matches_inclusive))
    #print('Time spent on qq-n-matches:', time() - t0)
    #return {**n_matches_exclusive, **n_matches_inclusive}, n_matches_by_level
    return {**n_matches_exclusive, **n_matches_inclusive}


def compute_n_matches_for_all_edges_with_codes(ilq: ILQuadTree, sp: SpatialPatternGraph):
    edges = sp.edges
    vertices = sp.vertices
    keywords = [v.keyword for v in vertices]
    depth = max([get_depth(ilq.quadtrees[keyword]) for keyword in keywords])
    #print('depth:', depth)
    # we need to reorder edges array to an optimal ordering to minimize computation efforts
    # 1) it partitions edges into two groups, where the first group
    # contains exclusive edges and the second group contains mutually
    # inclusive edges; 2) for each group, it ranks edges in an ascending
    # order of numbers of their n-matches in the previous level; and 3) by
    # concatenating edges in these two groups, it obtains the order of edges
    # for computing n-matches.
    exclusive_edges = [edge for edge in edges if edge.constraint['is_exclusive']]
    inclusive_edges = [edge for edge in edges if not edge.constraint['is_exclusive']]
    n_matches_exclusive = dict()
    n_matches_exclusive_codes = dict()
    previous_n_matches_exclusive = dict()
    previous_n_matches_exclusive_codes = dict()
    n_matches_inclusive = dict()
    n_matches_inclusive_codes = dict()
    previous_n_matches_inclusive = dict()
    previous_n_matches_inclusive_codes = dict()
    n_matches_codes_by_level = dict()
    for ee in exclusive_edges:
        wi, wj = ee.vi.keyword, ee.vj.keyword
        previous_n_matches_exclusive[ee] = [(ilq.quadtrees[wi], ilq.quadtrees[wj])]
        previous_n_matches_exclusive_codes[ee] = [('', '')]
        n_matches_codes_by_level[ee] = dict()
    for ie in inclusive_edges:
        wi, wj = ie.vi.keyword, ie.vj.keyword
        previous_n_matches_inclusive[ie] = [(ilq.quadtrees[wi], ilq.quadtrees[wj])]
        previous_n_matches_inclusive_codes[ie] = [('', '')]
        n_matches_codes_by_level[ie] = dict()
    candidate_nodes = dict()
    for level in range(1, max(2,depth+1)):
        for vertex in vertices:
            candidate_nodes[vertex] = set() # it is the set of nodes that are candidates to this vertex in this level
            
        for ee in exclusive_edges:
            n_matches_exclusive[ee], n_matches_exclusive_codes[ee] = compute_n_matches_at_level_with_codes(ilq, ee, level, previous_n_matches_exclusive[ee], previous_n_matches_exclusive_codes[ee], candidate_nodes[ee.vi], candidate_nodes[ee.vj])
            previous_n_matches_exclusive[ee] = n_matches_exclusive[ee]
            n_matches_codes_by_level[ee][level] = n_matches_exclusive_codes[ee]
            previous_n_matches_exclusive_codes[ee] = n_matches_exclusive_codes[ee]
            new_candidates_i, new_candidates_j = zip(*n_matches_exclusive[ee])
            if candidate_nodes[ee.vi]==set(): candidate_nodes[ee.vi] = set(new_candidates_i)
            else: candidate_nodes[ee.vi] = candidate_nodes[ee.vi].intersection(set(new_candidates_i))
            if candidate_nodes[ee.vj]==set(): candidate_nodes[ee.vj] = set(new_candidates_j)
            else: candidate_nodes[ee.vj] = candidate_nodes[ee.vj].intersection(set(new_candidates_j))
            
        for ie in inclusive_edges:
            n_matches_inclusive[ie], n_matches_inclusive_codes[ie] = compute_n_matches_at_level_with_codes(ilq, ie, level, previous_n_matches_inclusive[ie], previous_n_matches_inclusive_codes[ie], candidate_nodes[ie.vi], candidate_nodes[ie.vj])
            previous_n_matches_inclusive[ie] = n_matches_inclusive[ie]
            n_matches_codes_by_level[ie][level] = n_matches_inclusive_codes[ie]
            previous_n_matches_inclusive_codes[ie] = n_matches_inclusive_codes[ie]
            new_candidates_i, new_candidates_j = zip(*n_matches_inclusive[ie])
            if candidate_nodes[ie.vi]==set(): candidate_nodes[ie.vi] = set(new_candidates_i)
            else: candidate_nodes[ie.vi] = candidate_nodes[ie.vi].intersection(set(new_candidates_i))
            if candidate_nodes[ie.vj]==set(): candidate_nodes[ie.vj] = set(new_candidates_j)
            else: candidate_nodes[ie.vj] = candidate_nodes[ie.vj].intersection(set(new_candidates_j))
        
            
        # sort list  exclusive_edges according to len(n_matches_exclusive[ee])
        # also sort the list inclusive_edges according to len(n_matches_inclusive[ie])
        exclusive_edges.sort(key = lambda ee: len(n_matches_exclusive[ee]))
        inclusive_edges.sort(key = lambda ie: len(n_matches_inclusive[ie]))
        # print(level, 'edges with n-matches (exc,inc):', len(n_matches_exclusive), len(n_matches_inclusive))
        # print(level, 'total edges:', len(edges))
        # if len(n_matches_exclusive) + len(n_matches_inclusive) != len(edges):
        #     print(edges)
        #     print('---')
        #     print(n_matches_exclusive)
        #     print('---')
        #     print(n_matches_inclusive)
        
    #print('edges with n-matches (exc,inc):', len(n_matches_exclusive), len(n_matches_inclusive))
    n_matches = {**n_matches_exclusive, **n_matches_inclusive}
    return n_matches, n_matches_codes_by_level

def is_connected(vertex, vertices, edges):
    vertices_pairs = [(edge.vi, edge.vj) for edge in edges]
    vertices_pairs = list(filter(lambda vp: vertex in vp, vertices_pairs))
    for vp in vertices_pairs:
        if vp[0]==vertex and vp[1] in vertices:
            return True
        if vp[1]==vertex and vp[0] in vertices:
            return True
    return False

def find_skip_edges(edges_order):
    connected_vertices_subgraphs = []
    skip_edges = []
    for edge in edges_order:
        if not edge.constraint['is_exclusive']:
            for vertices_subgraph in connected_vertices_subgraphs:
                if {edge.vi, edge.vj}.issubset(vertices_subgraph):
                    skip_edges.append(edge)
                    break
        if connected_vertices_subgraphs==[]:
            connected_vertices_subgraphs.append({edge.vi, edge.vj})
        else:
            for i,vertices_subgraph in enumerate(connected_vertices_subgraphs):
                # find the subgraph that is connected (by some edge) to vi or vj, if there is any
                # if not, create a new subgraph for that edge
                if is_connected(edge.vi, vertices_subgraph, edges_order) or \
                    is_connected(edge.vj, vertices_subgraph, edges_order):
                    connected_vertices_subgraphs[i].add(edge.vi)
                    connected_vertices_subgraphs[i].add(edge.vj)
                    break
            # if connected_vertices_subgraphs wasn't empty but didn't have a connected subgraph to this edge, create a new subgraph
            connected_vertices_subgraphs.append({edge.vi, edge.vj})
    return skip_edges        

def compute_e_matches_for_an_edge(ilq: ILQuadTree, edge, n_matches_for_the_edge, candidate_objects = dict()):
    nodes_i, nodes_j = zip(*n_matches_for_the_edge)
    oss, ots = [], []
    for node_i in nodes_i:
        oss_ = node_i.nodes
        #if edge.id == 'hp' and len(oss_)>0:
        #    pass
        if candidate_objects_vi != set():
            oss_ = filter(lambda c: c.item in candidate_objects_vi, oss_)
        oss.extend(oss_)
    for node_j in nodes_j:
        ots_ = node_j.nodes
        if candidate_objects_vj != set():
            ots_ = filter(lambda c: c.item in candidate_objects_vj, ots_)
        ots.extend(ots_)
    # Now, we just need to analyse pairs of elements of (os X ot) to find e-matches
    e_matches = []
    for os in oss:
        for ot in ots:
            #print('I\'m here:', edge.id)
            if is_e_match(ilq, os.item, ot.item, edge):
                e_matches.append((os.item,ot.item))
    e_matches = list(set(e_matches))
    return e_matches

def compute_e_matches_for_an_edge2(ilq: ILQuadTree, edge, n_matches_for_the_edge, candidate_objects = dict()):
    #nodes_i, nodes_j = zip(*n_matches_for_the_edge)
    e_matches = []
    for node_i,node_j in n_matches_for_the_edge:
        oss = node_i.nodes
        if candidate_objects_vi != set():
            oss = filter(lambda c: c.item in candidate_objects_vi, oss)
        ots = node_j.nodes
        if candidate_objects_vj != set():
            ots = filter(lambda c: c.item in candidate_objects_vj, ots)
        for os in oss:
            for ot in ots:
                if is_e_match(ilq, os.item, ot.item, edge):
                    e_matches.append((os.item,ot.item))
    e_matches = list(set(e_matches))
    return e_matches

def find_sub_e_matches(n_match, edge, ilq, candidate_objects):
    #print('started running find_sub_e_matches')
    #t0 = time()
    e_matches = []
    node_i,node_j = n_match
    oss = [e.item for e in node_i.nodes]
    ots = [e.item for e in node_j.nodes]
    candidate_objects_vi = candidate_objects[edge.vi]
    candidate_objects_vj = candidate_objects[edge.vj]

    if candidate_objects_vi != set():
        # the intersection of children_i and candidate_nodes_vi
        oss = filter(set(candidate_objects_vi).__contains__, oss)
    if candidate_objects_vj != set():
        ots = filter(set(candidate_objects_vj).__contains__, ots)
    #print('Total oss, ots pairs:', len(oss), len(ots))
    for os in oss:
        for ot in ots:
            if is_e_match(ilq, os, ot, edge):
                e_matches.append((os,ot))
    #print('time spent on running find_sub_e_matches:', time()-t0)
    #print('ended running find_sub_e_matches')
    return e_matches

def compute_e_matches_for_an_edge_parallel(ilq: ILQuadTree, edge, n_matches_for_the_edge, candidate_objects = dict()):
    #print('started running compute_e_matches_for_an_edge_parallel')
    pool_obj = ThreadPool(int(multiprocessing.cpu_count()-1))
    #nodes_i, nodes_j = zip(*n_matches_for_the_edge)
    #t0 = time()
    find_sub_e_matches_partial = partial(find_sub_e_matches, edge = edge, ilq = ilq, candidate_objects = candidate_objects)
    
    # option 1: parallel
    #print('len(n_matches_for_the_edge):', len(n_matches_for_the_edge))
    #print('cumulatives during compute_e_matches_for_an_edge_parallel')
    #print('before pool map:', time()-t0)
    
    results = pool_obj.map(find_sub_e_matches_partial, n_matches_for_the_edge)
    #print('after pool map:', time()-t0)
    e_matches = set(itertools.chain(*results))
    #print('after itertools chain:', time()-t0)

    #option 2: sequential
    # print('chose sequencial (ematches)')
    # e_matches = []
    # for n_mt in n_matches_for_the_edge:
    #     e_matches.extend(find_sub_e_matches_partial(n_mt))

    
    #print('Time spent on compute_e_matches_for_an_edge_parallel:', time()-t0)
    #print('ended running compute_e_matches_for_an_edge_parallel')
    pool_obj.close()
    return e_matches

def compute_e_matches_for_all_edges(ilq: ILQuadTree, sp: SpatialPatternGraph, n_matches: dict, debug = True):
    #t0 = time()
    edges = sp.edges
    vertices = sp.vertices
    # we need to reorder edges array according to n_matches dictionary
    edges.sort(key = lambda e: len(n_matches[e]) or 0)
    skip_edges = find_skip_edges(edges)
    non_skip_edges = [e for e in edges if e not in skip_edges]
    e_matches = dict()
    
    candidate_objects = {vertex: set() for vertex in vertices} # it saves the set of objects that are candidates to each vertex 
    #print('Until end of pre-processing of e-matches computation:', time() - t0)
    for edge in non_skip_edges:
        if debug:
            print('- Computing e-matches for edge', edge.id)
        #print('start computing ematches for an edge')
        e_matches[edge] = compute_e_matches_for_an_edge_parallel(ilq, edge, n_matches[edge], candidate_objects)
        #print('ended computing ematches for an edge')
        if len(e_matches[edge])==0:
            return None, skip_edges, non_skip_edges
        if debug:
            print(f'- Total e-matches for edge {edge.id}: {len(e_matches[edge])}')
        candidate_objects_i, candidate_objects_j = zip(*e_matches[edge])
        if candidate_objects[edge.vi]==set(): candidate_objects[edge.vi] = set(candidate_objects_i)
        else: candidate_objects[edge.vi] = candidate_objects[edge.vi].intersection(set(candidate_objects_i))
        if candidate_objects[edge.vj]==set(): candidate_objects[edge.vj] = set(candidate_objects_j)
        else: candidate_objects[edge.vj] = candidate_objects[edge.vj].intersection(set(candidate_objects_j))

    #print('Time spent on e-matches:', time() - t0)
    return e_matches, skip_edges, non_skip_edges

def merge_partial_solutions(pa, pb, sp):
    # pa and pb and dictionaries in the format: {v1: obj1, ..., vn: objn} where vi's are vertices and obji's are GeoObjs
    # merging means aggregating the two partial solutions into a single one if possible
    # sometimes it's not possible, when the two solutions provide a different value for the same vertex
    merged = dict()
    for vertex in sp.vertices:
        if pa[vertex.id] is not None and pb[vertex.id] is not None and pa[vertex.id]!=pb[vertex.id]:
            return None # there is no merge (merging is impossible)
        merged[vertex.id] = pa[vertex.id] or pb[vertex.id] # becomes the one that is not the 'None' if there is one not being None
    return merged

def merge_lists_of_partial_solutions(pas, pbs, sp):
    merges_list = []
    for pa in pas:
        for pb in pbs:
            merge = merge_partial_solutions(pa, pb, sp)
            if merge is not None:
                merges_list.append(merge)
    return merges_list

def generate_partial_solutions_ematch(e_match, edge, sp):
    os, ot = e_match
    partial_solution = {vertex.id: None for vertex in sp.vertices}
    partial_solution[edge.vi.id] = os
    partial_solution[edge.vj.id] = ot
    return partial_solution

def generate_partial_solutions_from_e_matches_parallel(e_matches, edge, sp):
    pool_obj = ThreadPool(int(multiprocessing.cpu_count()-1))
    generate_partial_solutions_ematch_partial = partial(generate_partial_solutions_ematch, edge = edge, sp = sp)
    partial_solutions = pool_obj.map(generate_partial_solutions_ematch_partial, e_matches)
    #print('type(partial_solutions_ematch):', type(partial_solutions))
    #print('len:', len(partial_solutions))
    pool_obj.close()
    return partial_solutions

def filter_e_matches_by_vertex_candidates(e_matches, edge, candidates):
    #return [e for e in e_matches if (e[0] in candidates[edge.vi] and e[1] in candidates[edge.vj])]
    return list(filter(lambda e: (e[0] in candidates[edge.vi] and e[1] in candidates[edge.vj]), e_matches))
    
def join_e_matches2(sp: SpatialPatternGraph, e_matches: dict, n_matches: dict, skip_edges: list, non_skip_edges: list):
    #t0 = time()
    non_skip_edges.sort(key = lambda e: len(e_matches[e]))
    skip_edges.sort(key = lambda e: len(n_matches[e]))
    ordered_edges = non_skip_edges + skip_edges
    vertices = sp.vertices
    candidates = {vertex: set() for vertex in sp.vertices}
    for edge in non_skip_edges:
        cvi, cvj = zip(*e_matches[edge])
        if candidates[edge.vi] == set(): candidates[edge.vi] = set(cvi)
        else: candidates[edge.vi] = candidates[edge.vi].intersection(set(cvi))
        if candidates[edge.vj] == set(): candidates[edge.vj] = set(cvj)
        else: candidates[edge.vj] = candidates[edge.vj].intersection(set(cvj))
        
    # for vertex in candidates:
    #     candidates[vertex] = list(candidates[vertex])
        
    for edge in non_skip_edges:
        #print('filtering ematches by vertex candidates. Before:', len(e_matches[edge]), type(e_matches[edge]))
        #print('teste1')
        e_matches[edge] = filter_e_matches_by_vertex_candidates(e_matches[edge], edge, candidates)
        #print('teste2')
        #print('filtering ematches by vertex candidates. After:', type(e_matches[edge]))
        #print('len(emaches[edge]):', len(e_matches[edge]))
        
    partial_solutions = [{vertex.id: None for vertex in sp.vertices}]
    for edge in non_skip_edges:
        partial_solutions_edge = generate_partial_solutions_from_e_matches_parallel(e_matches[edge], edge, sp)
        #print('partial solutions edge[0]:', partial_solutions_edge[0])
        partial_solutions = merge_lists_of_partial_solutions(partial_solutions, partial_solutions_edge, sp)
        #print('partial solutions merged:', partial_solutions)
    #print('After generation of partial solutions:', len(partial_solutions))

    if None in partial_solutions:
        print('None is in partial solutions')
        print(sp)
    for edge in skip_edges:
        for i,solution in enumerate(partial_solutions):
            if solution is None:
                continue
            vi, vj = edge.vi, edge.vj
            lij, uij = edge.constraint['lij'], edge.constraint['uij']
            if solution is None:
                print('solution is None')
            if vi is None:
                print('Vi is None')
            if vj is None:
                print('Vj is None')
            os, ot = solution[vi.id], solution[vj.id]
            distance = os.distance(ot)
            if not(lij <= distance <= uij):
                partial_solutions[i] = None
    partial_solutions = filter(lambda x: x is not None, partial_solutions)

    final_solutions = []
    for solution in partial_solutions:
        solution_satisfy_qualitative_constraint = True
        for edge in sp.edges:
            if edge.constraint['relation'] is not None:
                vi, vj = edge.vi, edge.vj
                os, ot = solution[vi.id], solution[vj.id]
                if edge.constraint['relation'] != os.relation(ot):
                    solution_satisfy_qualitative_constraint = False
                    break
        if solution_satisfy_qualitative_constraint:
            final_solutions.append(solution)

    #print('Time spent on Joining:', time() - t0)
    return final_solutions


def QQ_SIMPLE(ilq: ILQuadTree, sp, debug = True):
    if sp.pattern_type == 'simple_graph':
        #pool_obj = ThreadPool(int(multiprocessing.cpu_count()-1))
        t0 = time()
        # uij_s = [edge.constraint['uij'] for edge in sp.edges if edge.constraint['relation'] is None]
        # if len(uij_s) == 0:
        #     max_uij = 1000
        # else:
        #     max_uij = max(uij_s)
        # for i, edge in enumerate(sp.edges):
        #     if edge.constraint['relation'] is not None and edge.constraint['relation'] != 'disjoint':
        #         sp.edges[i].constraint['lij'] = 0
        #         sp.edges[i].constraint['uij'] = max_uij
        keywords = [v.keyword for v in sp.vertices]
        if any([keyword not in ilq.quadtrees for keyword in keywords]):
            return [], time() - t0, psutil.Process().memory_info().rss/(2**20)
        if debug:
            print('Computing n-matches for edges')
        n_matches = compute_n_matches_for_all_edges(ilq, sp, debug = debug)
        if n_matches is None:
            return [], time() - t0, psutil.Process().memory_info().rss/(2**20)
        if debug:
            for edge in n_matches:
                print(f'- Total n-matches for edge {edge.id}: {len(n_matches[edge])}')
            print('Computing e-matches for edges')
        # if len(sp.edges) != len(n_matches):
        #     print('edges:',len(sp.edges))
        #     print('edges with nmatches:', len(n_matches))
        #     for edg in sp.edges:
        #         if edg not in n_matches:
        #             print(edg, 'not in nmatches')
        # avg_total_nmatches = 0
        # for edge in sp.edges:
        #     avg_total_nmatches += len(n_matches[edge])
        # avg_total_nmatches /= len(sp.edges)
        # print(f'avg_total_nmatches simple: {avg_total_nmatches}')
        
        e_matches, skip_edges, non_skip_edges = compute_e_matches_for_all_edges(ilq, sp, n_matches, debug)
        if e_matches is None:
            return [], time() - t0, psutil.Process().memory_info().rss/(2**20)
        else:
            # avg_total_ematches = 0
            # for edge in sp.edges:
            #     avg_total_ematches += len(e_matches[edge])
            # avg_total_ematches /= len(sp.edges)
            # print(f'avg_total_ematches simple: {avg_total_ematches}')
            
            if debug:
                print('Number of skip-edges:', len(skip_edges))
                print('Joining e-matches')
            solutions = join_e_matches2(sp, e_matches, n_matches, skip_edges, non_skip_edges)
            # solutions is a list of dictionaries in the format {v1: obj1, v2: obj2, ..., vn: objn} with matches to vertices 
        elapsed_time = time() - t0
        memory_usage = psutil.Process().memory_info().rss/(2**20)
        #pool_obj.close()
        return solutions, elapsed_time, memory_usage
    elif sp.pattern_type == 'Multi_keyword_vertices_graph':
        pool_obj = ThreadPool()
        t0 = time()
        qqespm_find_solutions_partial = partial(qqespm_find_solutions, ilquadtree = ilq)
        results = pool_obj.map(qqespm_find_solutions_partial, sp.spatial_patterns)
        all_solutions = list(itertools.chain(*results))
        elapsed_time = time() - t0
        memory_usage = psutil.Process().memory_info().rss/(2**20)
        pool_obj.close()
        return all_solutions, elapsed_time, memory_usage
            
        
def qqespm_find_solutions(ilquadtree, pattern):
    solutions, _, _ = QQESPM(ilquadtree, pattern)
    return solutions

def solutions_to_json(solutions, indent=None, only_ids = False):
    solutions_list = []
    for solution in solutions:
        if only_ids is False:
            solutions_list.append({vertex.id: solution[vertex_id].get_description() for vertex_id in solution})
        else:
            solutions_list.append({vertex.id: solution[vertex_id].item.get('osm_id') for vertex_id in solution})
    solutions_dict = {'solutions': solutions_list}
    return json.dumps(solutions_dict, indent=indent, ensure_ascii=False).encode('utf8').decode()


cache_circle_searches = {}

"""
from geoobject import GeoObj
from shapely.geometry import Point
from ilquadtree import ILQuadTree

geoobjs = [
    GeoObj(data = {'id':1}, keywords = ['school'], geometry = Point(0,0)),
    GeoObj(data = {'id':2}, keywords = ['house'], geometry = Point(1,1)),
    GeoObj(data = {'id':3}, keywords = ['pharmacy'], geometry = Point(2,2)),
    GeoObj(data = {'id':4}, keywords = ['house'], geometry = Point(10,10)),
]

ilquadtree = ILQuadTree(total_bbox = [-1,-1,11,11])
for geoobj in geoobjs:
    x, y = geoobj.geometry().centroid.coords[0]
    ilquadtree.insert(geoobj, keywords = geoobj.keywords)
    
# from espm import SpatialVertex, SpatialEdge, SpatialPatterGraph, ESPM

vertices = [
    SpatialVertex(1, 'house'),
    SpatialVertex(2, 'school'),
    SpatialVertex(3, 'pharmacy')
]

edges = [
    SpatialEdge(vertices[0], vertices[1], quantitative_constraint = {'lij':0, 'uij':2, 'sign':'<'}),
    SpatialEdge(vertices[0], vertices[2], quantitative_constraint = {'lij':0, 'uij':2, 'sign':'<'})                
]

sp = SpatialPatternGraph(vertices, edges)

ESPM(ilquadtree, sp)
"""
