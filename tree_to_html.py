from SREDT.SREDTClassifier import SREDTClassifier, SREDT_leaf, SREDT_node
from pyvis.network import Network
from networkx import DiGraph



def tree_to_html(clf: SREDTClassifier, feature_names: list[str], class_names: list[str]):
    graph = DiGraph()
    def node_to_html(node: SREDT_node | SREDT_leaf):
        if isinstance(node, SREDT_leaf):
            graph.add_node(id(node), label=node.__str__(labels=class_names) + node.details(), shape='box', color='lightblue')
            return id(node)

        graph.add_node(id(node), label=node.__str__(features=feature_names) + node.details(), shape='box', color='lightgreen')

        left_id = node_to_html(node.left)
        right_id = node_to_html(node.right)

        graph.add_edge(id(node), left_id, label='True', color='green')
        graph.add_edge(id(node), right_id, label='False', color='red')

        return id(node)
    
    node_to_html(clf.root_)
    net = Network(directed=True)
    net.set_options("""
    {
    "layout": {
        "hierarchical": {
        "direction": "UD",
        "sortMethod": "directed"
        }
    },
    "physics": {
        "hierarchicalRepulsion": {
        "nodeDistance": 300
        }
    }
    }
    """)
    net.from_nx(graph)

    return net.generate_html()

