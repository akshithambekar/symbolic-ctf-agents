import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_native_prefix_and_infix_are_bijectively_renamed():
    code='''
from extension.gold_benchmark import native_tree,render,check
assert render(native_tree('F & B F C','clean-up'),{'B':'p0','C':'p1'})=='F((p0) & (F(p1)))'
assert render(native_tree('F X','clean-up'),{'X':'p0'})=='F(p0)'
assert render(native_tree('~ ( purple_room ) U ( third_floor )','drone-planning'),{'purple_room':'p0','third_floor':'p1'})=='(!(p0)) U (p1)'
assert render(native_tree('G & U S ! A F A','pick-and-place'),{'S':'p0','A':'p1'})=='G(((p0) U (!(p1))) & (F(p1)))'
assert check('F(p0)','!G(!p0)')['outcome']=='equivalent'
assert check('F(p0)','G(F(p0))')['outcome']=='semantic_mismatch'
assert check('X(p0) | X(!p0)','p0 | !p0')['outcome']=='equivalent'
'''
    result=subprocess.run([sys.executable,'-c','import sys;sys.path.insert(0,"paper_benchmark");'+code],
                          cwd=ROOT.parent,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
