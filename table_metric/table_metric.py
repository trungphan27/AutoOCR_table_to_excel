# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.

from collections import deque

from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from rapidfuzz.distance import Levenshtein

from .parallel import parallel_process


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)


class CustomConfig(Config):
    def rename(self, node1, node2):
        if (
            node1.tag != node2.tag
            or node1.colspan != node2.colspan
            or node1.rowspan != node2.rowspan
        ):
            return 1.0
        if node1.tag == "td" and (node1.content or node2.content):
            return Levenshtein.normalized_distance(
                node1.content, node2.content
            )
        return 0.0


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (
                n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Convert an lxml table node into the tree format required by APTED."""
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for child in node.getchildren():
                self.load_html_tree(child, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        """Compute the TEDS score for one predicted/ground-truth HTML pair."""
        if not pred or not true:
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        try:
            pred_tree = html.fromstring(pred, parser=parser)
            true_tree = html.fromstring(true, parser=parser)
        except (etree.ParserError, ValueError):
            return 0.0
        pred_tables = pred_tree.xpath("body/table")
        true_tables = true_tree.xpath("body/table")
        if not pred_tables or not true_tables:
            return 0.0
        pred_table = pred_tables[0]
        true_table = true_tables[0]
        if self.ignore_nodes:
            etree.strip_tags(pred_table, *self.ignore_nodes)
            etree.strip_tags(true_table, *self.ignore_nodes)
        node_count = max(
            len(pred_table.xpath(".//*")), len(true_table.xpath(".//*"))
        )
        if node_count == 0:
            return 0.0
        distance = APTED(
            self.load_html_tree(pred_table),
            self.load_html_tree(true_table),
            CustomConfig(),
        ).compute_edit_distance()
        return 1.0 - float(distance) / node_count

    def batch_evaluate_html(self, pred_htmls, true_htmls):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
        '''
        if self.n_jobs == 1:
            scores = [self.evaluate(pred_html, true_html) for (
                pred_html, true_html) in zip(pred_htmls, true_htmls)]
        else:
            inputs = [{"pred": pred_html, "true": true_html} for (
                pred_html, true_html) in zip(pred_htmls, true_htmls)]

            scores = parallel_process(
                inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        return scores


if __name__ == '__main__':
    import json
    import pprint

    with open('sample_pred.json') as fp:
        pred_json = json.load(fp)
    with open('sample_gt.json') as fp:
        true_json = json.load(fp)
    teds = TEDS(n_jobs=4)
    scores = teds.batch_evaluate(pred_json, true_json)
    pp = pprint.PrettyPrinter()
    pp.pprint(scores)
