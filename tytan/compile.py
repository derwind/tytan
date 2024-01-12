import symengine
import numpy as np
import pandas as pd


def replace_function(expression, function, new_function):
    if expression.is_Atom:
        return expression
    else:
        replaced_args = (
                replace_function(arg, function,new_function)
                for arg in expression.args
            )
        if ( expression.__class__ == symengine.Pow):
            return new_function(*replaced_args)
        else:
            return expression.func(*replaced_args)


def highest_order(expanded_expr):
    max_exp = 0
    for k in expanded_expr.as_coefficients_dict():
        if k.is_Number:
            continue
        if k.is_Mul:
            exp = 0
            for k1 in k.args:
                if k1.is_Pow:
                    exp += k1.exp
                elif k1.is_Symbol:
                    exp += 1
            max_exp = max(max_exp, exp)
        elif k.is_Pow:
            max_exp = max(max_exp, k.exp)
    return max_exp


class Compile:
    def __init__(self, expr):
        self.expr = expr

    def get_qubo(self):
        """
        get qubo data
        Raises:
            TypeError: Input type is symengine, numpy or pandas.
        Returns:
            Tuple: qubo is dict. offset is float.
        """

        #symengine型のサブクラス
        if 'symengine.lib' in str(type(self.expr)):
            #式を展開して同類項をまとめる
            expr = symengine.expand(self.expr)

            #最高次数チェック
            def highest_order(expr):
                max_exp = 0
                for k in expr.as_coefficients_dict():
                    if k.is_Number:
                        continue
                    if k.is_Mul:
                        exp = 0
                        for k1 in k.args:
                            if k1.is_Pow:
                                exp += k1.exp
                            elif k1.is_Symbol:
                                exp += 1
                        max_exp = max(max_exp, exp)
                    elif k.is_Pow:
                        max_exp = max(max_exp, k.exp)
                return max_exp
            ho = highest_order(expr)
            if ho > 2:
                raise Exception(f'Error! The highest order of the constraint ({ho}) must be within 2!')

            #二乗項を一乗項に変換
            expr = replace_function(expr, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e)

            #もう一度同類項をまとめる
            expr = symengine.expand(expr)

            #定数項をoffsetとして抽出 #定数項は一番最後 #もう少し高速化できる？
            # offset = expr.as_ordered_terms()[-1] #定数項は一番最後 #もう少し高速化できる？
            # #定数項がなければ0
            # if '*' in str(offset):
            #     offset = 0
            offset = 0
            coeff_dict = expr.as_coefficients_dict()
            for ex, coeff in coeff_dict.items():
                if ex.is_Number:
                    offset = ex * coeff
                    break

            #offsetを引いて消す
            #expr2 = expr - offset

            #文字と係数の辞書
            #coeff_dict = expr2.as_coefficients_dict()

            #QUBO
            qubo = {}
            for key, value in coeff_dict.items():
                if key.is_Number:
                    continue
                tmp = str(key).split('*')
                #tmp = ['q0'], ['q0', 'q1']のどちらかになっていることを利用
                qubo[(tmp[0], tmp[-1])] = float(value)

            return qubo, offset

        # numpy
        elif isinstance(self.expr, np.ndarray):
            # 係数
            offset = 0

            # QUBOに格納開始
            qubo = {}
            for i, r in enumerate(self.expr):
                for j, c in enumerate(r):
                    if i <= j:
                        qubo[(f"q{i}", f"q{j}")] = c

            return qubo, offset

        # pandas
        elif isinstance(self.expr, pd.core.frame.DataFrame):
            # 係数
            offset = 0

            # QUBOに格納開始
            qubo = {}
            for i, r in enumerate(self.expr.values):
                for j, c in enumerate(r):
                    if i <= j and c != 0:
                        row_name, col_name = f"q{i}", f"q{j}"
                        if self.expr.index.dtype == "object":
                            row_name = self.expr.index[i]

                        if self.expr.columns.dtype == "object":
                            col_name = self.expr.columns[j]

                        qubo[(row_name, col_name)] = c

            return qubo, offset

        else:
            raise TypeError("Input type is symengine, numpy or pandas.")
