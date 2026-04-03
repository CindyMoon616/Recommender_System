import os
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

def batch_py_to_html(input_folder, output_folder):
    # 1. 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"❌ 错误：找不到输入文件夹 '{input_folder}'")
        return

    # 2. 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已创建输出文件夹: {output_folder}")

    # 3. 配置 HTML 格式化器
    # full=True: 生成完整的 HTML 页面（包含 CSS 样式）
    # style='monokai': 使用经典的黑底高亮主题
    # linenos=True: 显示行号
    formatter = HtmlFormatter(full=True, style='monokai', linenos=True)

    # 4. 遍历文件夹
    files = [f for f in os.listdir(input_folder) if f.endswith('.py')]
    
    if not files:
        print(f"🤔 在 '{input_folder}' 中没有找到任何 .py 文件。")
        return

    print(f"🚀 开始转换，共找到 {len(files)} 个文件...")

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        # 将文件名从 .py 替换为 .html
        output_filename = os.path.splitext(filename)[0] + ".html"
        output_path = os.path.join(output_folder, output_filename)

        try:
            # 读取代码
            with open(input_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # 生成高亮 HTML
            with open(output_path, 'w', encoding='utf-8') as f:
                highlight(code, PythonLexer(), formatter, outfile=f)
            
            print(f"✅ 已转换: {filename} -> {output_filename}")

        except Exception as e:
            print(f"❌ 转换 {filename} 时出错: {e}")

    print("\n✨ 全部任务处理完成！")

if __name__ == "__main__":
    # 配置文件夹路径
    SRC_DIR = "codes"
    DST_DIR = "pdf"
    
    batch_py_to_html(SRC_DIR, DST_DIR)