import akshare as ak
import pandas as pd
from tqdm import tqdm
import os
import time


class Stock:
    """
    股票类，存储股票基本信息及财务指标
    """
    def __init__(self, code, name):
        self.code = code
        self.name = name
        self.dv_ratio = None  # 股息率
        self.pe_ttm = None    # TTM市盈率
        self.pb = None        # 市净率
        self.total_mv = None  # 总市值
        
    def update_indicators(self, indicator_df):
        """更新财务指标"""
        self.dv_ratio = float(indicator_df["dv_ratio"].values[-1])
        self.pe_ttm = float(indicator_df["pe_ttm"].values[-1])
        self.pb = float(indicator_df["pb"].values[-1])
        self.total_mv = float(indicator_df["total_mv"].values[-1])


class RevertA:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.stock_em_df = None
        self.stock_zh_a_spot_em_df = None
        self.stock_zh_a_hist_df = None
        self.stock_zh_a_daily_df = None

    def get_all_stocks_with_cache(self, cache_file='all_stocks_cache.csv', cache_expire_seconds=86400):
        """
        获取所有股票代码，支持断点续传缓存。
        cache_file: 缓存文件名，默认'all_stocks_cache.csv'
        cache_expire_seconds: 缓存过期时间，默认一天（86400秒）
        """
        if os.path.exists(cache_file):
            # 检查缓存文件的修改时间
            last_modified = os.path.getmtime(cache_file)
            if time.time() - last_modified < cache_expire_seconds:
                # 缓存有效，直接读取
                try:
                    df = pd.read_csv(cache_file, index_col=0, dtype={'code': str})  # 强制股票代码为字符串
                    return df
                except Exception as e:
                    print(f"读取缓存文件失败，重新获取: {e}")

        # 缓存不存在或过期，重新获取
        df = ak.stock_info_a_code_name()
        # 保存到缓存文件
        df.to_csv(cache_file)
        return df

    def calculate_returns(self, stock_list):
        """
        计算股票列表中每只股票过去2年的收益率
        
        参数:
            stock_list: 股票代码列表
            
        返回:
            DataFrame: 包含股票代码和收益率的DataFrame
        """
        print("\n开始计算股票收益率...")
        try:
            end_date = pd.to_datetime(self.end_date)
            start_date = end_date - pd.DateOffset(years=2)
            print(f"计算区间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
            
            # 读取已处理的收益率数据
            processed_returns_codes = set()
            existing_returns_data = pd.DataFrame()
            
            # 读取已处理股票代码
            if os.path.exists("processed_returns_codes.txt"):
                try:
                    with open("processed_returns_codes.txt", "r") as f:
                        processed_returns_codes = set(f.read().splitlines())
                    print(f"已加载 {len(processed_returns_codes)} 个已处理股票的代码")
                except Exception as e:
                    print(f"警告: 读取已处理股票代码时出错: {e}")
            else:
                print("未找到已处理的股票代码文件，将处理所有股票")

            # 读取已存在的收益率数据
            if os.path.exists("returns_data.csv") and os.path.getsize("returns_data.csv") > 0:
                try:
                    existing_returns_data_original = pd.read_csv("returns_data.csv")
                    existing_returns_data = existing_returns_data_original.drop_duplicates()
                    print(f"已加载 {len(existing_returns_data)} 条历史收益率数据")
                    
                    if not existing_returns_data.empty and 'code' in existing_returns_data.columns:
                        existing_returns_data['code'] = existing_returns_data['code'].astype(str)
                        processed_returns_codes.update(existing_returns_data['code'].tolist())
                        print(f"已更新已处理股票代码，总计 {len(processed_returns_codes)} 个")
                except Exception as e:
                    print(f"警告: 读取收益率数据时出错: {e}")
                    existing_returns_data = pd.DataFrame()

            returns = []
            
            # 添加已存在的收益率数据
            if not existing_returns_data.empty:
                returns = existing_returns_data.to_dict('records')
                print(f"已添加 {len(returns)} 条历史收益率数据到结果集")

            # 过滤出未处理的股票
            unprocessed_stocks = [str(symbol) for symbol in stock_list if str(symbol) not in processed_returns_codes]
            print(f"需要处理 {len(unprocessed_stocks)} 只新股票")

            if not unprocessed_stocks:
                print("没有需要处理的新股票，使用缓存数据")
                returns_df = pd.DataFrame(returns)
                if returns_df.empty:
                    raise ValueError("没有可用的收益率数据")
                return returns_df

            # 处理新股票
            success_count = 0
            for symbol in tqdm(unprocessed_stocks, desc='计算收益率'):
                try:
                    # 确保symbol是字符串格式
                    symbol = str(symbol).rjust(6, '0')
                    hist_data = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                                 start_date=start_date.strftime('%Y%m%d'), 
                                                 end_date=end_date.strftime('%Y%m%d'), 
                                                 adjust="hfq")
                    
                    if hist_data is None or hist_data.empty or len(hist_data) < 2:
                        print(f"警告: 股票 {symbol} 历史数据不足或为空，跳过")
                        continue

                    if len(hist_data) < 400:
                        print(f"警告: 股票 {symbol} 历史数据不足400天，跳过")
                        continue

                    start_price = hist_data.iloc[0]['收盘']
                    end_price = hist_data.iloc[-1]['收盘']

                    # print(hist_data)
                    if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
                        print(f"警告: 股票 {symbol} 价格数据无效，跳过")
                        continue

                    total_return = (end_price - start_price) / start_price
                    returns.append({'code': symbol, 'return': total_return})
                    success_count += 1

                    pd.DataFrame(returns).to_csv("returns_data.csv", index=False)
                    with open("processed_returns_codes.txt", "a") as f:
                        for s in unprocessed_stocks[success_count-1:success_count]:
                            f.write(f"{s}\n")
                
                except Exception as e:
                    print(f"处理股票 {symbol} 时出错: {str(e)}")
                
            # 最终保存所有数据
            if returns:
                pd.DataFrame(returns).to_csv("returns_data.csv", index=False)
                with open("processed_returns_codes.txt", "a") as f:
                    for symbol in unprocessed_stocks[success_count - (success_count % 10):]:
                        f.write(f"{symbol}\n")
                print(f"\n成功处理 {success_count}/{len(unprocessed_stocks)} 只新股票的收益率数据")
            
            returns_df = pd.DataFrame(returns)
            if returns_df.empty:
                raise ValueError("没有成功计算任何股票的收益率")
                
            print(f"收益率计算完成，共 {len(returns_df)} 只股票")
            print(f"收益率统计:\n{returns_df['return'].describe()}")
            return returns_df
            
        except Exception as e:
            print(f"\n计算收益率时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 如果已有历史数据，则使用历史数据继续
            if 'returns_df' in locals() and not returns_df.empty:
                print("使用部分计算的收益率数据继续...")
                return returns_df
            else:
                # 尝试使用缓存的收益率数据
                try:
                    if os.path.exists("returns_data.csv"):
                        returns_df = pd.read_csv("returns_data.csv")
                        if not returns_df.empty:
                            print(f"使用缓存的收益率数据，共 {len(returns_df)} 条记录")
                            return returns_df
                except Exception as e:
                    print(f"读取缓存数据时出错: {str(e)}")
                raise ValueError("无法计算收益率且没有可用的缓存数据")

    def analyze_strategy(self):
        """
        实现股票策略分析：
        1. 筛选股息率>0的股票
        2. 计算过去2年收益率并分组
        3. 按市值分组
        4. 标记胜者/败者和市值大小
        
        返回:
            DataFrame: 包含分析结果的DataFrame，包含以下列：
            - code: 股票代码
            - name: 股票名称
            - return: 收益率
            - winner_loser: 胜者/败者标记
            - market_cap_group: 市值分组(small/medium/large)
            - pe_tag: PE分组标签
            - pb_tag: PB分组标签
        """
        # 读取已处理的股票代码和财务数据
        processed_codes = set()
        existing_data = pd.DataFrame()
        if os.path.exists("processed_codes_revertA.txt"):
            with open("processed_codes_revertA.txt", "r") as f:
                processed_codes = set(f.read().splitlines())
        
        if os.path.exists("dividend_stocks_data.csv"):
            existing_data = pd.read_csv("dividend_stocks_data.csv", dtype={'code': str})
            if not existing_data.empty and 'code' in existing_data.columns:
                processed_codes.update(existing_data['code'].astype(str).tolist())

        # 1. 获取有分红的股票(股息率>0)
        try:
            # 获取股票列表
            all_stocks = self.get_all_stocks_with_cache()
            stock_dict = {}
            for _, row in all_stocks.iterrows():
                if row['code'] not in processed_codes:
                    stock = Stock(row['code'], row['name'])
                    stock_dict[row['code']] = stock

            if not stock_dict:
                print("所有股票已处理完毕")
                
            # 获取财务指标数据
            dividend_stocks = []
            new_analysis_data = []
            
            # 添加已存在的数据
            if not existing_data.empty:
                new_analysis_data = existing_data.to_dict('records')
                
            for symbol, stock in tqdm(stock_dict.items(), desc='获取财务指标'):
                try:
                    if stock.code.rjust(6, '0')[:3] != '300' and stock.code.rjust(6, '0')[:3] != '688' and 'ST' not in stock.name and '退' not in stock.name:
                        indicator_df = ak.stock_a_indicator_lg(symbol=symbol)
                        stock.update_indicators(indicator_df)
                        print(f"处理股票: {stock.name}({stock.code}), 股息率: {stock.dv_ratio:.2f}%, PE(TTM): {stock.pe_ttm:.2f}, PB: {stock.pb:.2f}, 市值: {stock.total_mv/10000:.2f}亿")

                        # 记录已处理的股票
                        with open("processed_codes_revertA.txt", "a") as f:
                            f.write(f"{symbol}\n")
                        
                        if True:  # TODO: 添加条件判断
                            dividend_stocks.append(stock)
                            new_analysis_data.append({
                                'code': stock.code,
                                'name': stock.name,
                                'dividend_yield': stock.dv_ratio,
                                'pe_ttm': stock.pe_ttm,
                                'pb': stock.pb,
                                'total_mv': stock.total_mv
                            })

                        # 定期保存数据
                        pd.DataFrame(new_analysis_data).to_csv("dividend_stocks_data.csv", index=False)
                        
                except Exception as e:
                    print(f"Failed to get indicators for {symbol}: {str(e)}")
                    
            # 最终保存所有数据
            dividend_data = pd.DataFrame(new_analysis_data)
            dividend_data.to_csv("dividend_stocks_data.csv", index=False)
            
            if dividend_data.empty:
                raise ValueError("No dividend-paying stocks found in current batch")
                
            stock_list = dividend_data['code'].tolist()
            print(f"Found {len(stock_list)} dividend-paying stocks (including previously processed)")
            
            if len(stock_list) == 0:
                raise ValueError("No dividend-paying stocks found in current batch")
                
        except Exception as e:
            raise ValueError(f"Failed to get dividend data: {str(e)}")
        
        # 2. 计算过去2年收益率
        returns_df = self.calculate_returns(stock_list)
        
        # 按收益率分为10组
        try:
            returns_df = returns_df.dropna(subset=['return']).sort_values('return', ascending=False)
            returns_df['return_group'] = pd.qcut(returns_df['return'], q=10, labels=range(1, 11), duplicates='drop')
            returns_df['winner_loser'] = returns_df['return_group'].apply(
                lambda x: 'loser' if x == 1 else ('winner' if x == 10 else None)
            )
        except Exception as e:
            print(f"收益率分组时出错: {str(e)}")
            raise
        
        # 3. 直接合并收益率和股息数据
        print("\n合并收益率和股息数据...")
        try:
            # 确保code列类型一致
            returns_df['code'] = returns_df['code'].astype(str).str.strip().str.zfill(6)
            dividend_data['code'] = dividend_data['code'].astype(str).str.strip().str.zfill(6)
            
            # 合并数据
            merged_df = pd.merge(returns_df, dividend_data, on='code', how='left')
            
            # 检查合并结果
            if merged_df.empty:
                raise ValueError("合并后数据为空，可能是由于收益率数据和股息数据没有匹配项")
                
            # 记录合并前的行数
            before_drop = len(merged_df)
            
            # 删除缺失市值的数据
            merged_df = merged_df.dropna(subset=['total_mv'])
            
            # 记录删除的行数
            after_drop = len(merged_df)
            if before_drop > after_drop:
                print(f"警告: 移除了 {before_drop - after_drop} 条缺失市值的数据")
                
            print(f"合并后数据: {len(merged_df)} 条记录")
            
            # 检查是否有足够的股票进行分组
            if len(merged_df) < 10:
                raise ValueError(f"股票数量不足({len(merged_df)})，无法进行有意义的分组分析")
                
        except Exception as e:
            print(f"合并数据时出错: {str(e)}")
            raise
            
        # 4. 处理PE/PB分组
        print("\n开始处理PE/PB分组...")
        pe_df = dividend_data[['code', 'pe_ttm']].dropna()
        pe_df['pe_group'] = pd.qcut(pe_df['pe_ttm'], q=[0, 0.3, 0.7, 1],
                                  labels=['low', 'medium', 'high'])
        pe_df['pe_tag'] = pe_df['pe_group'].apply(lambda x: 'pe_low' if x == 'low' else ('pe_high' if x == 'high' else 'medium'))
        
        pb_df = dividend_data[['code', 'pb']].dropna()
        pb_df['pb_group'] = pd.qcut(pb_df['pb'], q=[0, 0.3, 0.7, 1],
                                  labels=['low', 'medium', 'high'])
        pb_df['pb_tag'] = pb_df['pb_group'].apply(lambda x: 'pb_low' if x == 'low' else ('pb_high' if x == 'high' else 'medium'))
        
        # 合并所有分组标签
        merged_df = pd.merge(merged_df, pe_df, left_on='code', right_on='code', how='left')
        merged_df = pd.merge(merged_df, pb_df, left_on='code', right_on='code', how='left')
        
        # 按总市值分组
        merged_df['market_cap_group'] = pd.qcut(merged_df['total_mv'], 
                                             q=[0, 0.3, 0.7, 1],
                                             labels=['small', 'medium', 'large'])
        
        return merged_df
    
    def get_small_cap_losers(self):
        """
        获取小市值败者股票(市值小且收益率最低的10%)
        返回: DataFrame包含小市值败者股票数据
        """
        df = self.analyze_strategy()
        print(df)
        # small_cap_losers = df[(df['market_cap_group'] == 'small') & (df['winner_loser'] == 'loser')]
        # return small_cap_losers
        
    def get_high_valuation_winners(self):
        """
        获取高估值胜者股票(PE高或PB高且收益率最高的10%)
        返回: DataFrame包含高估值胜者股票数据
        """
        df = self.analyze_strategy()
        # high_val_winners = df[(df['winner_loser'] == 'winner') &
        #                     ((df['pe_tag'] == 'pe_high') | (df['pb_tag'] == 'pb_high'))]
        # return high_val_winners


if __name__ == "__main__":
    start_date = '20230620'
    end_date = '20250620'
    revertA = RevertA(start_date, end_date)
    df = revertA.analyze_strategy()
    print(df)
    df.to_csv("revertA.csv")
    # small_cap_losers = revertA.get_small_cap_losers()
    # high_val_winners = revertA.get_high_valuation_winners()