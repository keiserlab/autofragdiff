import numpy as np
from rdkit import Chem
import networkx as nx
from prepare_fragments import *

def find_neigh_frags_to_neigh_atoms(bonds_broken_frags, bonds_broken_frag_ids):
    """ find a mapping between broken bonds (atom tuples) and their corresponding fragment ids (index tuple)
    bonds_broken_frags: -> tuple of bonds broken btween fragments
    bonds_broken_frag_ids -> ids of bonds broken between fragments
    """
    # a dictionary mapping from fragments (tuple) to atom ids (tuple) in the original moleucle
    neigh_frags_to_neigh_atoms = {}
    for i, bond in enumerate(bonds_broken_frags):
        neigh_frags_to_neigh_atoms[bonds_broken_frag_ids[i]] = bond
        neigh_frags_to_neigh_atoms[bonds_broken_frag_ids[i][::-1]] = bond[::-1] 
    
    return neigh_frags_to_neigh_atoms

class FragmentGraph():
    """ class for fragment graph
    """
    def __init__(self, mol, fragments, adjacency, frag_atom_ids, frag_to_id_dict, neigh_frags_to_neigh_atoms):
        self.original_mol = mol
        self.graph = nx.Graph()
        self.fragments = fragments
        self.frag_atom_ids = frag_atom_ids
        self.frag_to_id_dict = frag_to_id_dict
        self.conformer = mol.GetConformer().GetPositions() # position of atoms in the original molecule
        self.all_mol_atom_symbols = self.get_all_mol_atom_symbols() # array of atom symbols in the molecule
        self.all_mol_atom_charges = [] # including charge of atoms may not be wise since we are doint autoregressive generation
        self.neigh_frags_to_neigh_atoms = neigh_frags_to_neigh_atoms
        
        fragment_bonds = np.argwhere(np.triu(adjacency))
        
        for i, f in enumerate(fragments):
            self.graph.add_node(f, name='frag_' + str(i))
        
        for bond in fragment_bonds:
            self.graph.add_edge(fragments[bond[0]], fragments[bond[1]])
    
    def draw_graph(self, graph):
        labels = nx.get_node_attributes(graph, 'name')
        nx.draw_circular(graph, labels=labels, node_size=3000)
    
    def get_bfs_order(self, starting_point=None):
        """ returns a list of tuples of fragments that should be connected to traverse the graph
            in a BFS order
        """
        if starting_point is None:
            starting_point = 0
        starting_frag = self.fragments[starting_point]
        bfs_edges = list(nx.bfs_edges(self.graph, starting_frag))
        return bfs_edges

    def get_dfs_order(self, starting_point=None):
        """ return a list of tuples of fragments that should be connected to traverse
        in DFS order
        """
        if starting_point is None:
            starting_point = 0
        starting_frag = self.fragments[starting_point]
        dfs_edges = list(nx.dfs_edges(self.graph, starting_frag))
        return dfs_edges   

    def hierarchical_reconstruct(self, edge_order='BFS', starting_point=None):
        """ 
        Returns the reconstruction of the molecule in the order given in edge_order
        
        if edge_order is given the reconstruction is based on edge_order 
        else reconstructoin is based on BFS order with starting point
        
        Returns:
            hierarchical_mol : hierarchical molecule built in BFS order
            atom_ids_hierarchical: ids of atoms added in the BFS order 
        """
        if starting_point is None:
            starting_point = 0
            
        if edge_order == 'BFS':  
            edge_list = self.get_bfs_order(starting_point)
        elif edge_order == 'DFS':
            edge_list = self.get_dfs_order(starting_point)
        else:
            raise ValueError('edge order not found.')

        tmp = edge_list[0][0] # this is a mol
        tmp_id = self.frag_to_id_dict[tmp] # id of the fragment
        hierarchical_mol = [tmp] # the initial molecule 
        
        # ------------- find the atom ids in hier ------------
        atom_ids_hierarchical = [] # a set of atoms
        tmp_frag_atom_ids = self.frag_atom_ids[tmp_id]
        atom_ids_hierarchical.append(tmp_frag_atom_ids)
        
        
        # -------------- find the conformer in hier -----------
        hierarchical_conformer = [] # hierarchical conformeration of the molecule
        first_frag_conformer = self.transfer_conformer(tmp_frag_atom_ids)
        hierarchical_conformer.append(first_frag_conformer)
        
        
        # --------------- find the atom symbols in hier -------------
        hier_atom_symbol = []
        first_frag_symbols = self.all_mol_atom_symbols[list(tmp_frag_atom_ids)]
        hier_atom_symbol.append(first_frag_symbols)
        
        all_anchor_ids = []
        first_frag_id = tmp_id
        extensions_atom_ids = [self.frag_atom_ids[first_frag_id]]
        
        for edge in edge_list:
            
            tmp = Chem.CombineMols(tmp, edge[1])
            hierarchical_mol.append(tmp)
            frag_id = self.frag_to_id_dict[edge[1]] # id of the next fragment to add
            index_of_two_frags = (self.frag_to_id_dict[edge[0]], self.frag_to_id_dict[edge[1]])
            tmp_frag_atom_ids = tmp_frag_atom_ids.union(self.frag_atom_ids[frag_id])
            extensions_atom_ids.append(self.frag_atom_ids[frag_id])
            atom_ids_hierarchical.append(tmp_frag_atom_ids)
            
            anchor_idx = self.neigh_frags_to_neigh_atoms[index_of_two_frags][0]
            all_anchor_ids.append(anchor_idx)
    
            conformer_at_this_step = self.transfer_conformer(tmp_frag_atom_ids)
            hierarchical_conformer.append(conformer_at_this_step)
            
            hier_atom_symbol.append(self.all_mol_atom_symbols[list(tmp_frag_atom_ids)])
            
            
        return hierarchical_mol, atom_ids_hierarchical, extensions_atom_ids, hierarchical_conformer, hier_atom_symbol, all_anchor_ids
     
 
    def transfer_conformer(self, atom_ids):
        
        conformer_at_this_step = self.conformer[list(atom_ids)]
        return conformer_at_this_step

    def get_all_mol_atom_symbols(self):
        mol_atom_symbols = []
        for atom in self.original_mol.GetAtoms():
            mol_atom_symbols.append(atom.GetSymbol())
        mol_atom_symbols = np.array(mol_atom_symbols)
        return mol_atom_symbols
    
    def get_anchor_idx(self):
        pass

    
    @staticmethod
    def draw_fragment_graph(all_frags, fragment_bonds):
        
        G = nx.Graph()
        for i, f in enumerate(all_frags):
            img = Draw.MolToImage(f)
            img.save('frag_' + str(i)+ '.png')
        
        for i,f in enumerate(all_frags):
            G.add_node(f, name='frag_'+str(i), img=plt.imread('frag_' + str(i) + '.png'))
        
        for bond in fragment_bonds:
            G.add_edge(all_frags[bond[0]], all_frags[bond[1]])
        
        pos = nx.circular_layout(G)
        fig = plt.figure(figsize=(12,10))
        ax = plt.subplot(111)
        ax.set_aspect('equal')
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=2.)
        
        #plt.ylim(-4.5,4.5)
        trans=ax.transData.transform
        trans2=fig.transFigure.inverted().transform

        piesize=0.2 # this is the image size
        p2=piesize/2.0
        for n in G:
            xx,yy=trans(pos[n]) # figure coordinates
            xa,ya=trans2((xx,yy)) # axes coordinates
            a = plt.axes([xa-p2,ya-p2, piesize, piesize])
            a.set_aspect('equal')
            a.imshow(G.nodes[n]['img'])
            a.axis('off')
        ax.axis('off')
        plt.show()
        
    @staticmethod
    def draw_hier_recons(hier):
        graph = nx.Graph()
        for i, f in enumerate(hier):
            img = Draw.MolToImage(f)
            img.save('frag_' + str(i)+ '.png')
        
        for i, f in enumerate(hier):
            graph.add_node(f, name='frag_'+str(i), img=plt.imread('frag_' + str(i) + '.png'))

        for i in range(len(hier)-1):
            graph.add_edge(hier[i], hier[i+1])    
        
        pos = nx.circular_layout(graph, dim=2,scale=2.5)
        fig = plt.figure(figsize=(12,10))
        ax = plt.subplot(111)
        ax.set_aspect('equal')
        nx.draw_networkx_edges(graph, pos, ax=ax, node_size=40, edge_color='blue', width=1.5, arrows=True, arrowsize=30)


        #plt.ylim(-4.5,4.5)
        trans=ax.transData.transform
        trans2=fig.transFigure.inverted().transform

        piesize=0.22 # this is the image size
        p2=piesize/2.
        for n in graph:
            xx,yy=trans(pos[n]) # figure coordinates
            xa,ya=trans2((xx,yy)) # axes coordinates
            a = plt.axes([xa-p2,ya-p2, piesize, piesize])
            a.set_aspect('equal')
            a.imshow(graph.nodes[n]['img'])
            a.axis('off')
        ax.axis('off')
        plt.show()
        
    @staticmethod
    def get_smiles(mol):
        return Chem.MolToSmiles(mol)
    
    @staticmethod
    def get_mol_from_smiles(smiles):
        return Chem.MolFromSmiles(smiles)
    
    