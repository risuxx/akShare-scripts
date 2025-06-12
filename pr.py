import akshare as ak
import pandas as pd
from tqdm import tqdm
import os


class Stock:
    def __init__(self, code, name):
        self.average_roe = None
        self.dividend_per_share = None
        self.pe = None
        self.pe_ttm = None
        self.pb = None
        self.dividend_yield = None
        self.profit = None
        self.balance_ratio = None
        self.roe = None  # 净资产收益率（去除%后的数值）
        self.code = code
        self.name = name

    def get_financial_abstract(self):
        try:
            main_indict_df = ak.stock_financial_abstract_ths(self.code, "按年度")
            report_year = 2024
            if report_year in main_indict_df["报告期"].values:
                result = main_indict_df.loc[main_indict_df["报告期"] == report_year]
                roe_str = result["净资产收益率"].values[0]
                self.roe = float(roe_str.replace("%", "")) if roe_str else None
                self.balance_ratio = float(result["资产负债率"].values[0].replace("%", ""))
                if self.roe is None or self.roe < 10 or self.balance_ratio > 45:
                    return False
                roe_total = 0
                for i in range(0, 6):
                    if report_year - i in main_indict_df["报告期"].values:
                        roe_tmp_str = main_indict_df.loc[main_indict_df["报告期"] == report_year - i]["净资产收益率"].values[0]
                        roe_tmp = float(roe_tmp_str.replace("%", "")) if roe_tmp_str else None
                        if roe_tmp is None:
                            return False
                        roe_total += roe_tmp
                    else:
                        return False
                self.average_roe = roe_total / 6
                if self.average_roe * 2 < self.roe:
                    return False
            else:
                print(f"[-] 未找到{self.code}的2024年数据")
                return False
        except Exception as e:
            print(f"[-] 获取财务摘要失败，code:{self.code}, name:{self.name}")
            print(e)
            return False
        return True

    def get_stock_dividend_yield_pe(self):
        try:
            stock_price_df = ak.stock_a_indicator_lg(symbol=self.code)
            self.dividend_yield = float(stock_price_df["dv_ratio"].values[-1])
            self.pe_ttm = float(stock_price_df["pe_ttm"].values[-1])
            self.pe = float(stock_price_df["pe"].values[-1])
            self.pb = float(stock_price_df["pb"].values[-1])
            self.dividend_per_share = self.pe * self.dividend_yield
            dividend_yield_no_cnt = 0
            for i in range(10):
                # 一年大约250个股票交易日
                if float(stock_price_df["dv_ratio"].values[-i*250-1]) == 0 or stock_price_df["dv_ratio"].values[-i*250-1] is None:
                    dividend_yield_no_cnt += 1
                    return False
        except Exception as e:
            print(f"[-] 获取估值指标失败，code:{self.code}, name:{self.name}")
            print(e)
            return False
        return True

    def calc_pr(self):
        if any(v is None for v in [self.pb, self.average_roe, self.dividend_per_share]):
            return None
        if self.average_roe < 10 or self.dividend_per_share == 0:
            return None
        base_pr = ((self.pb / self.average_roe) / self.average_roe) * 100
        factor_a = 1 if self.dividend_per_share >= 50 else 50 / self.dividend_per_share
        pr = base_pr * factor_a
        return pr


def get_all_stock_code():
    # 读取已保存的结果
    if os.path.exists("sorted_data.csv"):
        existing_results = pd.read_csv("sorted_data.csv")
    else:
        existing_results = pd.DataFrame(columns=["code", "pr", "name"])

    # 读取已处理的股票代码
    processed_codes = set()
    if os.path.exists("processed_codes.txt"):
        with open("processed_codes.txt", "r") as f:
            processed_codes = set(f.read().splitlines())

    stock_info_df = ak.stock_info_a_code_name()

    for iiii, row in tqdm(stock_info_df.iterrows()):
        code = row["code"]
        name = row["name"]

        if code in processed_codes:
            continue  # 跳过已处理的股票

        try:
            # 记录已处理的股票（无论是否符合条件）
            with open("processed_codes.txt", "a") as f:
                f.write(f"{code}\n")
            stock = Stock(code, name)
            flag = stock.get_financial_abstract()
            if flag:
                if stock.get_stock_dividend_yield_pe() is False:
                    continue
                pr = stock.calc_pr()
                print(f"name:{name}, pr:{pr}, roe:{stock.roe}, average_roe:{stock.average_roe}")
                if pr is not None and pr < 0.5:
                    new_row = pd.DataFrame([[code, pr, name]], columns=["code", "pr", "name"])
                    existing_results = pd.concat([existing_results, new_row], ignore_index=True)
                    # 每处理10个股票保存一次
                    existing_results.to_csv("sorted_data.csv", index=False)
                    print(f"已保存至 sorted_data.csv（当前进度：{iiii+1}/{len(stock_info_df)}）")

        except Exception as e:
            print(f"[-] 处理 {code} 时发生错误：{str(e)}")
            # 即使出错也标记为已处理，避免重复尝试
            with open("processed_codes.txt", "a") as f:
                f.write(f"{code}\n")

    # 最终保存结果
    existing_results.to_csv("sorted_data.csv", index=False)
    print("所有数据处理完成！")


if __name__ == "__main__":
    get_all_stock_code()
    # stock = Stock("000661", "美的集团")
    # stock.get_stock_dividend_yield_pe()
    # stock.get_financial_abstract()
    # print(f"name:{stock.name}, pr:{stock.calc_pr()}, roe:{stock.roe}, average_roe:{stock.average_roe}")
    # stock_info_df = ak.stock_info_a_code_name()
    # stock_info_df.to_csv("stocks_name.csv", index=False)