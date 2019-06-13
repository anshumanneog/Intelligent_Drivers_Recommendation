
import pydotplus
from sklearn import tree
import collections
from sklearn.tree import _tree

class DecisionTreeVisualization:
    ''' To visualize Decision Tree and get PDF output
    '''
    def __init__(self,data,list_cols,DV,classifier):
        ''' Initialization 
        :param data : Training Data with Dependentt and Independent Variables
        :param classifier : Define Decision Trees classifier with appropriate parameters
        :param list_cols: Names of independent Variables
        :param DV : Dependent variable
        :return: None
        '''
        self.data = data
        self.list_cols = list_cols
        self.DV = DV
        self.classifier = classifier
        
    def DecisionTree(self):
        '''
        Train the Decision Tree
        :return: Classifier Object
        '''
        X_train = self.data[self.list_cols]
        Y_train = self.data[self.DV]
        clf = self.classifier.fit(X_train,Y_train)
        return(clf)
        
    def TreeGraph(self): 
        ''' To visulaize the Decision Tree
        :return: graph
        '''
        try:
            dot_data = tree.export_graphviz(decision_tree=self.DecisionTree(), feature_names=self.list_cols,
                                            out_file=None,class_names=None,
                                            label='all', filled=True,leaves_parallel=False, 
                                            impurity=True, node_ids=False,proportion=False, rotate=False,
                                            rounded=True,special_characters=False,  precision=3 )
    
            graph = pydotplus.graph_from_dot_data(dot_data)
            
            colors = ('turquoise', 'orange')
            edges = collections.defaultdict(list)
            
            for edge in graph.get_edge_list():
                edges[edge.get_source()].append(int(edge.get_destination()))
            
            for edge in edges:
                edges[edge].sort()    
                for i in range(2):
                    dest = graph.get_node(str(edges[edge][i]))[0]
                    dest.set_fillcolor(colors[i])
            return(graph)
        except:
            print('Error')
    ## Convert tree in terms of code:    
    def tree_to_code(self):
    	'''
    	Outputs a decision tree model as a Python function
    	:param tree: decision tree model
    		The decision tree to represent as a function
    	:param feature_names: list
    		The feature names of the dataset used for building the decision tree
    	'''
    	tree_ = self.DecisionTree().tree_
#       feature_names = self.list_cols
    	feature_name = [	self.list_cols[i] if i != _tree.TREE_UNDEFINED else "undefined!"	for i in tree_.feature]
    	#print("def tree({}):".format(", ".join(feature_names)))
    
    	def recurse(node, depth):
    		indent = "  " * depth
    		if tree_.feature[node] != _tree.TREE_UNDEFINED:
    			name = feature_name[node]
    			threshold = tree_.threshold[node]
    			print("{}if {} <= {}:".format(indent, name, threshold))
    			recurse(tree_.children_left[node], depth + 1)
    			print("{}else:  # if {} > {}".format(indent, name, threshold))
    			recurse(tree_.children_right[node], depth + 1)
    		else:
    			print("{}return {}".format(indent, tree_.value[node]))
    
    	recurse(0, 1)

       
       
