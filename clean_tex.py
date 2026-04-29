with open('面试复盘/项目/项目.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到第二个\end{document}的位置
first_end = content.find(r'\end{document}')
if first_end != -1:
    second_end = content.find(r'\end{document}', first_end + 1)
    if second_end != -1:
        # 只保留到第二个\end{document}
        cleaned_content = content[:second_end + len(r'\end{document}')]
        with open('面试复盘/项目/项目.tex', 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        print('文件已清理完成')
    else:
        print('只找到一个end{document}')
else:
    print('未找到end{document}')
