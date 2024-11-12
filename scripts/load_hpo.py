from pathlib import Path
from scripts.graph_library import Dag
from pronto import Ontology 

def create_dag(file):
    print("Creating Dag with {}".format(file))
    file = Path(file)
    dag = Dag()

    def add_keys(key1, key2):
        dag.append(key1)
        dag.append(key2)
        dag.set_node_key(key1)
        dag.add_node_child (key2, key1)
        dag.add_node_parent(key1, key2)  

    if file.suffix == '.csv':
        try:
            with open(file) as f:
                next(f)
                for line in f:
                    x = line.split(',')
                    key1, key2 = x[0:2]
                    add_keys(key1, key2)
            
        except:
            pass

    if file.suffix == '.obo':
        ## pronto.Ontology  v0.12 or lower than 2 - use the following code

        # obo = Ontology(file)
        # for term in obo: #pronto.Ontology v2: obo -> obo.terms()
        #     for parent in term.parents: #pronto.Ontology v2: term.parents -> term.superclass()
        #         add_keys(term.id, parent.id)
        # for term in obo: #pronto.Ontology v2: obo -> obo.terms()
        #     dag.set_name(term.id, term.name)
            
        ## pronto.Ontology V2:
        obo = Ontology(file)
        for term in obo.terms():
            for parent in term.superclasses(with_self=False, distance=1):
                add_keys(term.id, parent.id)
        for term in obo.terms():
            dag.set_name(term.id, term.name)
            
    print("Created dag using HPO Codes n = {}".format(len(dag)))
    return dag