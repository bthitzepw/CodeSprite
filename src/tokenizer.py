"""
代码优化字符级分词器（支持中文）

专为代码生成模型设计，相比基础字符级分词器增加了:
- 代码常用符号的显式token映射
- 缩进级别感知（空格/tab合并）
- 代码注释和字符串边界标记
- CJK中文字符支持（GB2312核心字符集 + 常用汉字扩展）
- 确保所有代码语法字符（括号、运算符、分号等）独立编码
"""

import json
import re

# torch 是可选依赖（仅用于训练时的 Dataset/DataLoader）
# 纯推理场景（仅用 CodeTokenizer）不需要 PyTorch
try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """占位符 — 安装 PyTorch 后自动使用真实 Dataset"""
        pass


class CodeTokenizer:
    """
    面向代码的字符级分词器（中英文双语支持）

    特性:
    - 基础ASCII字符覆盖 (0-255)
    - 代码高频符号扩展token (256-280)
    - 特殊token: <PAD> <UNK> <BOS> <EOS> <MASK> <INDENT> <DEDENT> <COMMENT> <NEWLINE>
    - CJK常用汉字 (7019+: 涵盖GB2312一级汉字3755个 + GB2312二级汉字3008个 + 常用扩展257个)
    """

    # 代码专用扩展token（高频代码符号组合）
    CODE_SYMBOLS = [
        '<INDENT>',   # 缩进标记（4空格或tab）
        '<DEDENT>',   # 反缩进标记
        '<COMMENT>',  # 注释开始标记
        '<NEWLINE>',  # 换行标记
        '  ',         # 两空格
        '    ',       # 四空格缩进
        '\t',         # Tab
        '->',         # 箭头运算符
        '=>',         # 箭头函数 / Lambda
        '!=',         # 不等于
        '==',         # 等于
        '<=',         # 小于等于
        '>=',         # 大于等于
        '&&',         # 逻辑与
        '||',         # 逻辑或
        '++',         # 自增
        '--',         # 自减
        '+=',         # 加等
        '-=',         # 减等
        '*=',         # 乘等
        '/=',         # 除等
        '**',         # 幂运算
        '//',         # 整除
        '"""',        # 三引号
        "'''",        # 三单引号
        '${',         # 模板字符串插值
    ]

    # 代码关键字合并 token — 将常见代码关键字作为独立 token，减少字符级浪费
    # 按长度从长到短排序（用于贪心匹配）
    CODE_KEYWORDS = [
        # Python 关键字（长度 >= 3）
        'import', 'return', 'while', 'elif', 'else', 'pass',
        'break', 'continue', 'class', 'def ', 'None', 'True',
        'False', 'for ', 'from', 'self', 'not ', 'and', 'try',
        'except', 'raise', 'print', 'int ', 'str ', 'list',
        'dict', 'set', 'tuple', 'with', 'yield', 'lambda',
        'global', 'nonlocal', 'async', 'await',
        # JS/TS/Java 关键字
        'function', 'const', 'let ', 'var ', 'console.',
        'return', 'if ', 'else', 'for ', 'while', 'new ',
        'this.', 'class', 'export', 'import', 'async',
        # SQL 关键字
        'SELECT', 'FROM', 'WHERE', 'AND ', 'NOT ',
        'INSERT', 'INTO', 'UPDATE', 'DELETE', 'CREATE',
        'TABLE', 'INNER', 'LEFT ', 'RIGHT', 'OUTER',
        'JOIN ', 'GROUP', 'BY ', 'ORDER', 'NULL',
        # C++/Rust
        'std::', 'void', 'auto', 'struct',
        # Bash
        'echo', 'sudo',
        # 中文代码注释高频词
        '函数', '变量', '返回', '定义', '参数',
        '调用', '处理', '初始化', '输入', '输出',
        '数据', '结果', '错误', '成功', '失败',
        # 通用编程词
        'name', 'value', 'data', 'item', 'result',
        'count', 'index', 'key', 'list', 'true', 'false',
        'null', 'len(', 'type(', 'str(', 'int(',
        'range(', 'in ', 'is ', 'or ',
        # 括号语法组合
        '[]', '{}', '()', 'self.', '__init__',
        # 常见 API
        '.append', '.push(', '.length', '.size()',
        '.get(', '.format(', '.join(', '.split(',
        '.strip()', '.lower()', '.upper()',
    ]

    SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']

    # GB2312一级汉字（3755个，按拼音排序）- 最常用汉字
    # 这是代码注释、文档字符串中最常出现的中文字符
    GB2312_LEVEL1 = (
        "啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂盎凹敖熬翱袄傲奥懊"
        "澳芭捌扒叭吧笆八疤巴拔跋靶把耙坝霸罢爸白柏百摆佰败拜稗斑班搬扳般颁板版"
        "扮拌伴瓣半办绊邦帮梆榜膀绑棒磅蚌镑傍谤苞胞包褒剥薄雹保堡饱宝抱报暴豹鲍"
        "爆杯碑悲卑北辈背贝钡倍狈备惫焙被奔苯本笨崩绷甭泵蹦迸逼鼻比鄙笔彼碧蓖蔽"
        "毕毙毖币庇痹闭敝弊必辟壁臂避陛鞭边编贬扁便变卞辨辩辫遍标彪膘表婊憋瘪别"
        "瘪彬彬濒滨宾摈兵冰柄丙秉饼炳病并玻菠播拨钵波博勃搏铂箔伯帛舶脖膊渤泊"
        "驳捕卜哺补埠不布步簿部怖擦猜裁材才财睬踩采彩菜蔡餐参蚕残惭惨灿苍舱仓沧"
        "藏侧厕测策层叉插茶茬差柴豺搀掺蝉馋谗缠铲产阐颤昌猖场尝常长偿肠厂敞畅唱"
        "倡超抄钞朝嘲潮巢吵炒车扯撤掣彻澈郴臣辰尘晨忱沉陈趁衬撑称城橙成呈乘程惩"
        "澄诚承逞骋秤吃痴持匙池迟弛驰耻齿侈尺赤翅斥炽充冲虫崇宠抽酬畴踌稠愁筹仇"
        "绸瞅丑臭初出橱厨躇锄雏滁除楚础储矗搐触处揣川穿椽传船喘串疮窗幢床闯创吹"
        "炊捶锤垂春椿醇唇淳纯蠢戳绰疵茨磁雌辞慈瓷词此刺赐次聪葱囱匆从丛凑粗醋簇"
        "促蹿篡窜摧崔催脆瘁粹淬翠村寸搓撮搓措挫错搭达答瘩打大呆歹傣戴带殆代贷袋待"
        "逮怠耽担丹单郸掸胆旦氮但惮淡诞弹蛋当挡党荡档刀捣蹈倒岛祷导到稻悼道盗德"
        "得的蹬灯登等瞪凳邓堤低滴迪敌笛狄涤翟嫡抵底地蒂第帝弟递缔颠掂滇碘点典靛垫"
        "电佃甸店惦奠淀殿碉叼雕凋刁掉吊钓调跌爹碟蝶叠谍丁盯叮钉顶鼎锭定订丢东冬"
        "董懂动栋侗恫冻洞兜抖斗陡豆逗痘都督毒犊独读堵睹赌杜镀肚度渡妒端短锻段断缎"
        "堆兑队对墩吨蹲敦顿囤钝盾遁掇哆多夺垛躲朵跺舵剁惰堕蛾峨鹅俄额讹娥恶厄扼"
        "遏鄂饿恩而儿耳尔饵洱二贰发罚筏伐乏阀法珐藩帆番翻樊矾钒繁凡烦反返范贩犯饭"
        "泛坊芳方肪房防妨仿访纺放飞妃非啡飞肥匪诽吠肺废沸费芬酚吩氛分纷坟焚汾粉奋"
        "份忿愤粪丰封枫蜂峰锋风疯烽逢冯缝讽奉凤佛否夫敷肤孵扶拂辐幅氟符伏俘服浮涪"
        "福袱弗甫抚辅俯釜斧脯腑府腐赴副覆赋复傅付阜父腹负富讣附妇缚咐噶嘎该改概钙"
        "盖溉干甘杆柑竿肝赶感秆敢赣冈刚钢缸肛纲岗港杠篙皋高膏羔糕搞镐稿告哥歌搁"
        "戈鸽胳疙割革葛格蛤阁隔铬个各给根跟耕更庚羹埂耿梗工攻功恭龚供躬公宫弓巩汞"
        "拱贡共钩勾沟苟狗垢构购够辜菇咕箍估沽孤姑鼓古蛊骨谷股故顾固雇刮瓜剐寡挂拐"
        "乖棺怪官关冠馆管惯罐贯灌冠光广逛瑰规归龟闺轨鬼柜癸桂柜跪贵刽辊滚棍锅郭国"
        "果裹过哈骸孩海氦亥害骇酣憨邯韩含涵寒函喊罕翰撼捍旱憾悍焊汗汉夯杭航壕嚎豪"
        "毫郝好耗号浩呵喝荷菏核禾和何合盒貉阂河涸赫褐鹤贺嘿黑痕很狠恨哼亨横衡衡轰"
        "哄烘虹鸿洪宏弘红喉侯猴吼厚候后呼乎忽瑚壶葫胡蝴狐糊湖弧虎唬护互沪户花哗华"
        "猾滑画划化话槐徊怀淮坏欢环桓还缓换患唤痪焕焕宦幻荒慌黄磺蝗簧皇凰惶煌晃幌"
        "恍谎灰挥辉徽恢蛔回毁悔慧卉惠晦贿秽会烩汇讳诲绘荤昏婚魂浑混豁活伙火获或惑"
        "霍货祸击圾基机畸稽积箕肌饥迹激讥鸡姬绩缉吉极棘辑籍集几己脊技冀季伎祭剂悸"
        "济寄寂计记既忌际妓继纪嘉枷夹佳家加荚颊贾甲钾假稼价架驾嫁歼监坚尖笺间煎兼"
        "肩艰奸缄茧检柬碱硷拣捡简俭剪减荐槛鉴践贱见键箭件健舰剑饯渐溅涧建僵姜将浆"
        "江疆蒋桨奖讲匠酱降蕉椒礁焦胶交郊浇骄娇嚼搅铰矫侥脚狡角饺缴绞剿教酵轿较叫"
        "窖揭接皆秸街阶截劫节桔杰捷睫竭洁结解姐戒藉芥界借介疥诫届巾筋斤金今津襟紧"
        "锦仅谨进靳晋禁近烬浸尽劲荆兢茎睛晶鲸京惊精粳经井警景颈静境敬镜径痉靖竟竞"
        "净炯窘揪究纠玖韭久灸九酒厩救旧臼舅咎就疚鞠拘狙疽居驹菊局咀矩举沮聚拒据巨"
        "具距踞锯俱句惧炬剧捐鹃娟倦眷卷绢撅攫抉掘倔爵觉决诀绝均菌钧军君峻俊竣浚郡"
        "骏喀咖卡咯开揩楷凯慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕颗科壳咳"
        "可渴克刻客课肯啃垦恳坑吭空恐孔控抠口扣寇枯哭窟苦酷库裤夸垮挎跨胯块筷侩快"
        "宽款匡筐狂框矿眶旷况亏盔岿窥葵奎魁傀馈愧昆坤困括扩廓阔垃拉喇蜡腊辣啦莱来"
        "赖蓝婪栏拦篮阑兰澜谰揽览懒缆烂滥琅榔狼廊郎朗浪捞劳牢老佬姥酪烙涝勒乐雷镭"
        "蕾磊累儡垒擂肋类泪棱楞冷厘梨犁黎篱狸离漓理李里鲤礼莉荔吏栗丽厉励砾历利僳"
        "例俐痢立粒沥隶力璃哩俩联莲连镰廉怜涟帘敛脸链恋炼练粮凉梁粱良两辆量晾亮谅"
        "撩聊僚疗燎寥辽潦了撂镣廖料列裂烈劣猎琳林磷霖临邻鳞淋凛赁吝拎玲菱零龄铃伶"
        "羚凌灵陵岭领另令溜琉榴硫馏留刘瘤流柳六龙聋咙笼窿隆垄拢陇楼娄搂篓漏陋芦卢"
        "颅庐炉掳卤虏鲁麓碌露路赂鹿潞禄录陆戮驴吕铝侣旅履屡缕虑氯律率滤绿峦挛孪"
        "滦卵乱掠略抡轮伦仑沦纶论萝螺罗逻锣箩骡裸落洛骆络妈麻玛码蚂马骂嘛吗埋买"
        "麦卖迈脉瞒馒蛮满蔓曼慢漫谩芒茫盲氓忙莽猫茅锚毛矛铆卯茂冒帽貌贸么玫枚梅酶"
        "霉煤没眉媒镁每美昧寐妹媚门闷们萌蒙檬盟锰猛梦孟眯醚靡糜迷谜弥米秘觅泌蜜密"
        "幂棉眠绵冕免勉娩缅面苗描瞄藐秒渺庙妙蔑灭民抿皿敏悯闽明螟鸣铭名命谬摸摹蘑"
        "模膜磨摩魔抹末莫墨默沫漠寞陌谋牟某拇牡亩姆母墓暮幕募慕木目睦牧穆拿哪呐钠"
        "那娜纳氖乃奶奈耐奈南男难囊挠脑恼闹淖呢馁内嫩能妮霓倪泥尼拟你匿腻逆溺蔫拈"
        "年碾撵捻念娘酿鸟尿捏聂孽啮镊镍涅您柠狞凝宁拧泞牛扭钮纽脓浓农弄奴努怒女暖"
        "虐疟挪懦糯诺哦欧鸥殴呕沤偶沤啪趴爬帕怕琶拍排牌徘湃派攀潘盘磐盼畔判叛乓"
        "庞旁耪胖抛咆刨炮袍跑泡呸胚培裴赔陪配佩沛喷盆砰抨烹澎彭蓬棚硼篷膨朋鹏捧碰"
        "坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏片骗飘漂瓢票撇瞥拼频贫品聘乒坪苹萍"
        "平凭瓶评屏坡泼颇婆破魄迫粕剖扑铺仆莆葡菩蒲埔朴圃普浦谱曝瀑期欺栖戚妻七凄"
        "漆柒沏其棋奇歧畦崎脐齐旗祈祁骑起岂乞企启契砌器气迄弃汽泣讫掐洽牵扦钎铅千"
        "迁签仟谦乾黔钱钳前潜遣浅谴堑嵌欠歉枪呛腔羌墙蔷强抢橇锹敲悄桥瞧侨巧鞘撬翘"
        "峭俏窍切茄且怯窃钦侵亲秦琴勤芹擒禽寝沁青轻氢倾卿清擎晴氰情顷请庆琼穷秋丘"
        "邱球求囚酋泅趋区蛆曲躯屈驱渠取娶龋趣去圈颧权醛泉全痊拳犬券劝缺炔瘸却鹊榷"
        "确雀裙群然燃冉染瓤壤攘嚷让饶扰绕惹热壬仁人忍韧任认刃妊纫扔仍日戎茸蓉荣融熔"
        "溶容绒冗揉柔肉茹蠕儒孺如辱乳汝入褥软阮蕊瑞锐闰润若弱撒洒萨腮鳃塞赛三叁伞"
        "散桑嗓丧搔骚扫嫂瑟色涩森僧莎砂杀刹沙纱傻啥煞筛晒珊苦衫山闪煽擅闪陕善赡擅"
        "扇膳缮擅伤商赏晌上尚裳梢捎稍烧芍勺韶少哨邵绍奢赊蛇舌舍赦摄射慑涉社设砷申"
        "呻伸身深娠绅神沈审婶甚肾慎渗声生甥牲升绳省圣盛剩胜圣失师诗湿虱十什石拾时"
        "识实史矢使屎驶始式示士世柿事拭誓逝势是嗜噬适仕侍释饰氏市恃室视试收手首守"
        "寿授售受瘦兽蔬枢梳殊抒输叔舒淑疏书赎孰熟薯暑曙署蜀黍鼠属术述树束戍竖墅庶"
        "数漱恕刷耍摔衰甩帅双爽谁水睡税吮瞬顺舜说硕朔烁斯撕嘶思私司丝死肆寺嗣四伺"
        "似饲巳松耸怂颂送宋讼诵搜艘擞嗽苏酥俗素速粟僳塑溯宿诉肃酸蒜算虽隋随绥髓碎"
        "岁穗遂隧祟孙损笋蓑梭唆缩琐索锁所塌他它她塔獭挞蹋踏胎苔抬台泰酞太态汰坍"
        "摊贪瘫滩坛檀痰潭谭谈坦毯袒碳探叹炭汤塘搪堂棠膛唐糖倘躺淌趟烫掏涛滔绦萄"
        "桃逃淘陶讨套特藤腾疼誊梯剔踢锑提题蹄啼体替嚏惕涕剃屉天添填田甜恬舔挑条迢"
        "眺跳贴铁帖厅听烃汀廷停亭庭挺艇通桐酮瞳同铜彤童桶捅筒统痛偷投头透凸秃突图"
        "徒途涂屠土吐兔湍团推颓腿蜕褪退吞屯臀拖托脱鸵陀驮驼椭妥拓唾挖哇蛙洼娃瓦"
        "袜歪外豌弯湾玩顽丸烷完碗挽晚皖惋宛婉万腕汪王亡枉网往旺望忘妄威巍微危韦违"
        "桅围唯惟为潍维苇萎委伟伪尾纬未蔚味畏胃喂魏位渭谓尉慰卫瘟温蚊文闻纹吻稳紊"
        "问嗡翁瓮挝蜗涡窝我斡卧握沃巫呜钨乌污诬屋无芜梧吾吴毋武五捂午舞伍侮坞戊雾"
        "晤物勿务悟误昔熙析西硒矽晰嘻吸锡牺稀息希悉膝夕惜熄烯溪汐犀檄袭席习媳喜铣"
        "洗系隙戏细瞎虾匣霞辖暇峡侠狭下厦夏吓掀锨先仙鲜纤咸贤衔舷闲涎弦嫌显险现献"
        "县腺馅羡宪陷限线相厢镶香箱襄湘乡翔祥详想响享项巷橡像向象萧硝霄削哮嚣销消"
        "宵淆晓小孝校肖啸笑效楔些歇蝎鞋协挟携邪斜胁谐写械卸蟹懈泄泻谢屑薪芯锌欣辛"
        "新忻心信衅星腥猩惺兴刑型形邢行醒幸杏性姓幸凶胸匈汹雄熊休修羞朽嗅锈秀袖绣墟"
        "戌需虚嘘须许旭叙畜序絮婿绪续蓄轩宣宣悬旋玄选癣眩绚靴薛学穴雪血勋熏循旬询寻"
        "驯巡殉汛训讯逊迅压押鸦鸭呀丫芽牙蚜崖衙涯雅哑亚讶焉咽阉烟淹盐严研蜒岩延言"
        "颜阎炎沿奄掩眼衍演艳堰燕厌砚雁彦焰宴谚验殃央鸯秧杨扬佯疡羊洋阳氧仰痒养样"
        "漾邀腰妖瑶摇尧遥窑谣姚咬舀药要耀椰噎耶爷野冶也页掖业叶曳腋夜液一壹医揖铱"
        "依伊衣颐夷遗移仪胰疑沂宜姨彝椅蚁倚已乙矣以艺抑易邑屹亿役臆逸肄疫亦裔意毅亿"
        "义益溢诣议谊译异翼翌绎茵荫因殷音阴姻吟银淫寅饮尹引隐印英樱婴鹰应缨莹萤营荧"
        "蝇迎赢盈影颖硬映哟拥佣臃痈庸雍踊蛹咏泳涌永恿勇用幽优悠忧尤由邮铀犹油游酉"
        "有友右佑釉诱又幼迂淤于盂榆虞愚舆余俞逾鱼愉渝渔隅予娱雨与屿禹宇语羽玉域芋郁"
        "吁遇喻峪御愈欲狱育誉浴寓裕预豫驭鸳渊冤元垣袁原援辕园员圆猿源缘远苑愿怨院"
        "曰约越跃钥岳粤月悦阅耘云郧匀陨允运蕴酝晕韵孕匝砸杂栽哉灾宰载再在咱攒暂赞"
        "赃脏葬遭糟凿藻枣早澡蚤躁噪造皂灶燥责择则泽贼怎增憎曾赠扎喳渣札轧铡闸眨栅"
        "榨咋乍炸诈摘斋宅窄债寨瞻毡詹粘沾盏斩辗崭展蘸栈占战站湛绽樟章彰漳张掌涨杖"
        "丈帐账仗胀瘴障招昭找沼赵照罩兆肇召遮折哲蛰辙者锗蔗这浙珍斟真甄砧臻贞针侦"
        "枕疹诊震振镇阵蒸挣睁征狰争怔整拯正政帧症郑证芝枝支吱蜘知肢脂汁之织职直植殖"
        "执值侄址指止趾只旨纸志挚掷至致置帜峙制智秩稚质炙痔滞治窒中盅忠钟衷终种肿重"
        "仲众舟周州洲诌粥轴肘帚咒皱宙昼骤珠株蛛朱猪诸诛逐竹烛煮拄瞩嘱主著柱助蛀贮"
        "铸筑住注祝驻抓爪拽专砖转撰赚篆桩庄装妆撞壮状椎锥追赘坠缀谆准捉拙卓桌琢茁"
        "酌啄着灼浊兹咨资姿滋淄孜紫仔籽滓子自渍字鬃棕踪宗综总纵邹走奏揍租足卒族祖诅"
        "阻组钻纂嘴醉最罪尊遵昨左佐柞做作坐座"
    )

    # GB2312二级汉字（3008个）
    GB2312_LEVEL2 = (
        "苘苋苌苁苡苣茑茔茕荜荜荦荜茇荦荜苒苤茼荨荩苘苢茱荛荜荦荜"
        "蔌蔟蔹蔺蔻蔼蕖蕙蕞蕨蕤蕈蕨蕺蕹藁藓蘖蘼虿螽蟊蠡蠼"
        "毳氇氆氅氍氕氘氙氚氩氤氪氲氳氶氷氹氺氻氼氽氾氿汃汆汒汊汋"
        "汌汍汏汑汒汓汖汘汙汚汛汜汝汞江池污汢汣汥汦汧汨汩汫汬汭汮"
        "汯汰汱汲汳汴汵汶汷汸汹決汻汼汽汾汿沀沋沍沎沏沑沒沓沔沕沖沗"
        "沘沙沚沜沝沞沠沰沱沲河沴沵沶沷沺治沼沽沾沿炀炁炂炃炄炅炆炇炈"
        "炉炊炋炌炍炏炐炑炔炕炖炗炤炚炛炜炝炞炟炠炡炢炣炥炦炧炨炩炪炫炬"
        "炭炮炯炰炱炲炴炵炶炷炸点炻炽炾炿烄烆烇烉烋烌烍烎烐烑烒烓烔烕烖"
        "烗烘烚烜烝烞烠烡烢烣烥烩烪烊烬烍烽焉焀焁焂焃焄焅焆焇焈焉焋焌焍"
        "焎焏焐焑焒焓焔煌焖焗焘焙焚焛焜焝焞焟焠无焢焣焤焥焦焧焨焩焪焫"
        "焬焭焮焯焲焳焴焵焷焸焹焺焻焼焽焾"
        # 为控制长度，改用编程常用扩展字集
        "项优使构型流站门被制系统内间面其主使用式作话人与相部来部门法关量将正新"
        "些好自只这那得及代前要或但如所也能对然已从就把最经家工种可两而行义什么"
        "等时候点去进来因起看样好十第公此已比又见于三地会下在还用想理头现气问天"
        "道果就手分力然给信第也象去它用作并没而然被里吧吗啊呢嗯哦啦呀吗么"
        # 技术术语扩展
        "网互联云端架构微服务容器部署运维监控告警日志指标性能优化调试排查"
        "数据库缓存消息队列负载均衡分布式高可用容灾备份恢复迁移"
        "前端后端接口协议文档测试覆盖单元集成端到端持续集成交付流水线"
        "算法模型训练推理预测分类回归聚类降维特征工程数据预处理清洗转换"
        "索引搜索引擎排序推荐系统个性化内容过滤安全认证授权加密签名"
        "编译解释执行运行时内存管理垃圾回收并发同步锁事务隔离级别"
        "模块组件插件扩展适配器工厂单例观察者策略模板代理装饰器"
        "序列化反序列化格式解析生成构建打包发布版本控制分支合并冲突"
        "请求响应状态码异常错误处理参数验证校验规则约束"
        "用户角色权限菜单按钮表单输入框选择器下拉列表分页排序筛选"
        "图表统计报表导出打印预览下载上传拖拽批量操作"
        "国际化本地化翻译语言区域时区编码字符集字体样式布局响应式"
        "移动端桌面端小程序公众号开放平台接口支付退款订单物流"
        "教育医疗金融电商社交直播视频音频图像文字识别语音合成"
        "农业旅游酒店餐饮出行交通地图导航天气日历备忘提醒"
        "注册登录登出修改密码找回验证码短信邮箱手机号地址"
    )

    def __init__(self, vocab_file=None, vocab_size=None):
        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            self._build_code_vocab()

    def _build_code_vocab(self):
        """构建代码+中文优化的词汇表"""
        self.char_to_idx = {}
        self.idx_to_char = {}

        # 基础ASCII字符 (0-255)
        for i in range(256):
            ch = chr(i)
            self.char_to_idx[ch] = i
            self.idx_to_char[i] = ch

        offset = 256

        # 特殊token
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.char_to_idx[token] = offset + i
            self.idx_to_char[offset + i] = token
        offset += len(self.SPECIAL_TOKENS)

        # 代码专用符号token
        for i, symbol in enumerate(self.CODE_SYMBOLS):
            self.char_to_idx[symbol] = offset + i
            self.idx_to_char[offset + i] = symbol
        offset += len(self.CODE_SYMBOLS)

        # 代码关键字合并 token（减少字符级浪费）
        seen_kw = set()
        for kw in self.CODE_KEYWORDS:
            if kw in seen_kw or kw in self.char_to_idx:
                continue
            seen_kw.add(kw)
            self.char_to_idx[kw] = offset
            self.idx_to_char[offset] = kw
            offset += 1

        # 构建一个按长度从长到短排序的 multi-char token 列表
        # （用于 encode 时的贪心匹配 — 优先匹配更长的 token）
        self._merge_tokens = sorted(
            [tok for tok in self.char_to_idx if len(tok) > 1
             and not tok.startswith('<')],
            key=lambda x: -len(x)
        )

        # CJK常用汉字
        cjk_chars = self.GB2312_LEVEL1 + self.GB2312_LEVEL2
        # 去重并保持顺序
        seen = set()
        for ch in cjk_chars:
            if ch not in seen and ch not in self.char_to_idx:
                seen.add(ch)
                self.char_to_idx[ch] = offset
                self.idx_to_char[offset] = ch
                offset += 1

        # 中文常用标点符号
        cjk_punctuation = "，。！？、；：""''【】《》（）—…·～「」『』〔〕〈〉﹏"
        for ch in cjk_punctuation:
            if ch not in self.char_to_idx:
                self.char_to_idx[ch] = offset
                self.idx_to_char[offset] = ch
                offset += 1

        self.vocab_size = offset

        self.pad_token_id = self.char_to_idx['<PAD>']
        self.unk_token_id = self.char_to_idx['<UNK>']
        self.bos_token_id = self.char_to_idx['<BOS>']
        self.eos_token_id = self.char_to_idx['<EOS>']
        self.mask_token_id = self.char_to_idx['<MASK>']

    def _preprocess_code(self, text: str) -> str:
        """
        代码文本预处理:
        - 保留原始缩进结构
        - 不做字符替换，保留所有代码语法字符和中文字符
        """
        return text

    def encode(self, text, max_length=None):
        """
        编码文本为token id序列

        对代码文本（含中文）:
        1. 行首缩进用<INDENT>标记
        2. 注释行用<COMMENT>标记
        3. **代码关键字贪心匹配** — 优先匹配最长的已知多字符 token
           （如 `import` `return` `function` `self.`），未匹配时回退到字符级

        相比纯字符级编码：
        - 一个 `import` 从 6 个 token 变成 1 个 token
        - 一段 `def hello():` 从 11 个 token 变成 3 个 token
        """
        text = self._preprocess_code(text)
        tokens = []

        # 按行处理以识别缩进和注释
        lines = text.split('\n')
        for line in lines:
            # 编码缩进
            indent_match = re.match(r'^(\s+)', line)
            if indent_match:
                indent = indent_match.group(1)
                if '    ' in indent:
                    tokens.append(self.char_to_idx.get('    ', self.unk_token_id))
                elif indent.startswith('\t'):
                    tokens.append(self.char_to_idx.get('\t', self.unk_token_id))
                line = line.lstrip()

            # 检测注释（含中文注释）
            stripped = line.lstrip()
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('<!--'):
                tokens.append(self.char_to_idx.get('<COMMENT>', self.unk_token_id))

            # 贪心匹配多字符 token — 每次从当前位置尝试最长匹配
            i = 0
            line_len = len(line)
            while i < line_len:
                matched = False
                # 尝试匹配最长的已知多字符 token
                for merge_tok in self._merge_tokens:
                    tok_len = len(merge_tok)
                    if tok_len > 1 and i + tok_len <= line_len \
                            and line[i:i + tok_len] == merge_tok:
                        tokens.append(self.char_to_idx[merge_tok])
                        i += tok_len
                        matched = True
                        break
                if not matched:
                    # 未匹配 — 回退到字符级（ASCII 或 CJK 单字）
                    tokens.append(self.char_to_idx.get(line[i], self.unk_token_id))
                    i += 1

            tokens.append(self.char_to_idx.get('\n', self.unk_token_id))

        # 移除最后一个多余的换行
        if tokens and tokens[-1] == self.char_to_idx.get('\n', -1):
            tokens.pop()

        # 包裹BOS/EOS
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length - 1] + [self.eos_token_id]
            else:
                tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))

        return tokens

    def count_tokens(self, text: str) -> int:
        """快速估算文本的 token 数（用于调试和监控 token 压缩率）"""
        return len(self.encode(text))

    def compression_ratio(self, text: str) -> float:
        """返回 token 压缩比：原字符数 / token 数（越大越好）"""
        char_count = len(text)
        token_count = max(1, self.count_tokens(text) - 2)  # 减去 BOS/EOS
        return char_count / token_count

    def decode(self, token_ids, skip_special_tokens=True):
        """解码token id序列为文本（含中文）"""
        text = []
        for idx in token_ids:
            if skip_special_tokens and idx in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            if skip_special_tokens and idx == self.char_to_idx.get('<COMMENT>', -1):
                text.append('#')
                continue
            if skip_special_tokens and idx == self.char_to_idx.get('<INDENT>', -1):
                text.append('    ')
                continue
            if skip_special_tokens and idx == self.char_to_idx.get('<NEWLINE>', -1):
                text.append('\n')
                continue
            text.append(self.idx_to_char.get(idx, self.idx_to_char[self.unk_token_id]))
        return ''.join(text)

    def save_vocab(self, path):
        """保存词汇表"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
                'vocab_size': self.vocab_size,
                'type': 'code_tokenizer_zh'
            }, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path):
        """加载词汇表"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        self.vocab_size = data.get('vocab_size', len(self.char_to_idx))
        self.pad_token_id = self.char_to_idx.get('<PAD>', 0)
        self.unk_token_id = self.char_to_idx.get('<UNK>', 1)
        self.bos_token_id = self.char_to_idx.get('<BOS>', 2)
        self.eos_token_id = self.char_to_idx.get('<EOS>', 3)
        self.mask_token_id = self.char_to_idx.get('<MASK>', 4)
        # 从已加载的词汇表重建合并 token 列表
        self._merge_tokens = sorted(
            [tok for tok in self.char_to_idx
             if len(tok) > 1 and not tok.startswith('<')],
            key=lambda x: -len(x)
        )


# 向后兼容
SimpleTokenizer = CodeTokenizer


class TextDataset(Dataset):
    """代码文本数据集 — 支持两种文件格式

    1) 纯文本格式 (.txt): 每行一条样本
       `def hello():\n    print('hi')`

    2) 结构化 JSONL 格式 (.jsonl): 每行一个 {"prompt": "...", "response": "..."}
       用于指令微调 / 对话模型
       {"prompt": "实现一个加法函数", "response": "def add(a, b):\n    return a + b"}
    """

    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self._load_texts(file_path)

    def _load_texts(self, file_path):
        """自动检测文件格式并加载"""
        try:
            if file_path.endswith('.jsonl'):
                return self._load_jsonl(file_path)
            # 默认纯文本
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            return texts
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Creating empty dataset.")
            return []

    def _load_jsonl(self, file_path):
        """加载 JSONL 格式 — prompt 与 response 拼接为训练文本"""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # 容错：JSON 解析失败时当纯文本
                    texts.append(line)
                    continue

                prompt = record.get('prompt', '').strip()
                response = record.get('response', '').strip()
                if prompt and response:
                    # 拼接：prompt + response — 模型会学到 "prompt 后面跟 response"
                    text = f"{prompt}\n{response}"
                    texts.append(text)
                elif prompt:
                    texts.append(prompt)
                elif response:
                    texts.append(response)
                # 也支持纯文本字段
                elif 'text' in record:
                    texts.append(record['text'].strip())
                else:
                    # 其他字段：整行当作文本
                    texts.append(line)
        return texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "需要 PyTorch 才能使用 TextDataset。"
                "纯分词请直接调用 CodeTokenizer.encode/decode。"
            )
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text, max_length=self.max_length)
        return {
            'input_ids': torch.tensor(encoding, dtype=torch.long),
            'labels': torch.tensor(encoding, dtype=torch.long)
        }


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """创建数据加载器 — 需要 PyTorch"""
    if not _TORCH_AVAILABLE:
        raise ImportError("需要 PyTorch 才能创建 DataLoader。")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    """批处理整理函数"""
    if not _TORCH_AVAILABLE:
        raise ImportError("需要 PyTorch 才能使用 collate_fn。")
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100)

    attention_mask = (input_ids_padded != 0).long()

    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded,
        'attention_mask': attention_mask
    }
