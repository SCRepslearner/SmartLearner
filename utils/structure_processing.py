import os
import solidity_parser
from solidity_parser import parser
from pprint import pprint
from treelib import Tree


# value_map = {"FunctionDefinition": "name", "Parameter": "name", "Identifier": "name", "BinaryOperation": "operator",
#              "NumberLiteral": "number", }


def get_type_value(node):
    value = None
    if 'name' in node:
        value = node['name']
        value_name = 'name'
    elif 'value' in node:
        value = node['value']
        value_name = 'value'
    elif 'number' in node:
        value = node['number']  # NumberLiteral
        value_name = 'number'
    elif 'namePath' in node:
        value = node['namePath']
        value_name = 'namePath'
    elif 'operator' in node:
        value = node['operator']
        value_name = 'operator'
    elif 'functionName' in node:
        value = node['functionName']
        value_name = 'functionName'
    elif 'memberName' in node:
        value = node['memberName']
        value_name = 'memberName'
    elif 'initialValue' in node:
        value = node['initialValue']
        value_name = 'initialValue'
    elif 'default' in node:
        value = node['default']
        value_name = 'default'
    elif 'libraryName' in node:
        value = node['libraryName']
        value_name = 'libraryName'
    else:
        if node['type'] not in ['ParameterList', 'Block', 'ExpressionStatement', 'FunctionCall', 'TupleExpression',
                                'VariableDeclarationStatement', 'IndexAccess', 'IfStatement', 'AssemblyBlock',
                                'AssemblyAssignment', 'ElementaryTypeNameExpression', 'Mapping', 'ForStatement',
                                'ArrayTypeName', 'InLineAssemblyStatement', 'AssemblySwitch', 'AssemblyLocalDefinition',
                                'AssemblyIf', 'WhileStatement', 'Conditional', 'NewExpression', 'EmitStatement',
                                'InheritanceSpecifier', 'AssemblyFor', 'DoWhileStatement']:
            # pprint(node)
            pass
        value_name = None

    node_type = node['type']
    if value is None:
        value = node['type'] + "Value"

    return node_type, value, value_name


class StructureProcessing:
    def __init__(self, path):
        self.path = path
        self.cnt = 0
        self.trees = []
        self.samples = []

    def recursive_fn(self, tree, parent_idf, subtree):
        for key, value in subtree.items():
            if value is None:
                self.cnt += 1
                tree.create_node(tag=key, identifier=self.cnt, data=value, parent=parent_idf)
            elif isinstance(value, str):
                self.cnt += 1
                tree.create_node(tag=key, identifier=self.cnt, data=value, parent=parent_idf)
            elif isinstance(value, bool):
                self.cnt += 1
                tree.create_node(tag=key, identifier=self.cnt, data=str(value), parent=parent_idf)
            elif isinstance(value, dict):
                if 'type' in value:
                    self.cnt += 1
                    node_type, node_value, value_name = get_type_value(value)
                    tree.create_node(tag=node_type, identifier=self.cnt, data=node_value, parent=parent_idf)
                    cur_idf = self.cnt
                    del value['type']
                    if value_name is not None:
                        del value[value_name]
                    self.recursive_fn(tree, cur_idf, value)
                else:
                    self.recursive_fn(tree, self.cnt, value)

            elif isinstance(value, list):
                if len(value) == 0:
                    self.cnt += 1
                    tree.create_node(tag=key, identifier=self.cnt, data=key + "Value", parent=parent_idf)
                else:
                    for child in value:  # a dict
                        if not isinstance(child, dict):
                            # print('child:', child)
                            continue
                        if 'type' in child:
                            self.cnt += 1
                            node_type, node_value, value_name = get_type_value(child)
                            tree.create_node(tag=node_type, identifier=self.cnt, data=node_value, parent=parent_idf)
                            del child['type']
                            if value_name is not None:
                                del child[value_name]
                            cur_idf = self.cnt
                            self.recursive_fn(tree, cur_idf, child)
                        else:
                            self.recursive_fn(tree, self.cnt, child)

            else:
                continue
                # print("Unprocessed Type", type(value), key, value)

    def generate_trees(self):
        try:
            source_unit = parser.parse_file(self.path)
            # pprint(source_unit)
        except:
            # print('[ERROR]Parse solidity file failure：', (os.path.basename(self.path)))
            return
        if 'children' not in source_unit:
            return
        for child in source_unit['children']:
            if child is None:
                continue
            if child['type'] != 'ContractDefinition':
                continue
            else:
                tree = Tree()
                self.cnt = 0
                tree.create_node(tag=child['type'], identifier=self.cnt, data=child['name'], parent=None)
                root_idf = self.cnt
                del child['type']
                del child['name']
                self.recursive_fn(tree, root_idf, child)
                # tree.show()
                self.trees.append(tree)

    def pre_order_traversal(self):
        for tree in self.trees:
            val_seq = []
            type_seq = []
            for idf in tree.expand_tree():
                node = tree.get_node(idf)
                type_seq.append(node.tag)
                val_seq.append(str(node.data))
            sample = {"type": type_seq, "value": val_seq}
            self.samples.append(sample)


# if __name__ == '__main__':
#     file_path = 'test.sol.sol'
#     processor = StructureProcessing(file_path)
#     processor.generate_trees()
#     processor.pre_order_traversal()
#     for sample in processor.samples:
#         print(sample['type'])
#         print(sample['value'])
