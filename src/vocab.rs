//! 词表：零样本标签（闭集）+ 中→英映射词典。可持续扩充。
//!
//! 本文件由 tools/export_vocab.py 从 Python 版 src/vocab.py 生成，改词表请改那边再重新导出。
//! 生成器保证折行只发生在词条之间 —— 字符串字面量内绝不会出现 `\n`。

/// 零样本标签闭集（138 个），CLIP 文本塔用 `a photo of {tag}` 编码
pub static TAG_VOCAB: &[&str] = &[
    "beach", "seaside", "mountain", "snowy mountain", "forest", "lake", "river", "waterfall",
    "sunset", "sunrise", "night", "starry sky", "city skyline", "city street", "park", "bridge",
    "field", "garden", "swimming pool", "temple", "church", "museum", "hotel room", "bedroom",
    "kitchen", "living room", "office", "classroom", "restaurant", "cafe", "bar", "supermarket",
    "playground", "amusement park", "zoo", "aquarium", "concert", "cinema", "stadium", "airport",
    "train station", "airplane", "boat", "train", "camping tent", "ski resort", "hiking trail",
    "rainy", "foggy", "sunny", "cloudy", "snowing", "indoor", "outdoor", "selfie", "portrait",
    "group photo", "child", "baby", "man", "woman", "family", "couple", "wedding",
    "birthday party", "graduation", "crowd", "dress", "skirt", "shirt", "suit", "uniform",
    "swimsuit", "hat", "sunglasses", "glasses", "scarf", "down jacket", "wedding dress",
    "christmas tree", "dog", "cat", "puppy", "kitten", "bird", "fish", "horse", "rabbit", "deer",
    "butterfly", "panda", "elephant", "camel", "food", "dessert", "cake", "fruit", "hotpot",
    "barbecue", "noodles", "coffee", "milk tea", "wine", "beer", "car", "bicycle", "motorcycle",
    "bus", "truck", "laptop", "television", "camera", "book", "newspaper", "toy", "balloon",
    "flower bouquet", "roses", "luggage", "lantern", "fireworks", "document", "invoice", "receipt",
    "ticket", "boarding pass", "menu", "poster", "whiteboard", "screenshot", "chat screenshot",
    "id card", "passport", "banknote", "coin", "painting", "calligraphy", "map",
];

pub static ZH2TAG: &[(&str, &str)] = &[
    ("生日蛋糕", "cake"), ("毕业典礼", "graduation"), ("聊天记录", "chat screenshot"),
    ("游泳池", "swimming pool"), ("博物馆", "museum"), ("办公室", "office"), ("咖啡店", "cafe"),
    ("咖啡馆", "cafe"), ("游乐场", "amusement park"), ("动物园", "zoo"), ("海洋馆", "aquarium"),
    ("水族馆", "aquarium"), ("演唱会", "concert"), ("音乐会", "concert"), ("电影院", "cinema"),
    ("体育馆", "stadium"), ("火车站", "train station"), ("高铁站", "train station"), ("全家福", "family"),
    ("很多人", "crowd"), ("鸭舌帽", "hat"), ("太阳镜", "sunglasses"), ("羽绒服", "down jacket"),
    ("圣诞树", "christmas tree"), ("自行车", "bicycle"), ("摩托车", "motorcycle"), ("公交车", "bus"),
    ("笔记本", "laptop"), ("行李箱", "luggage"), ("登机牌", "boarding pass"), ("身份证", "id card"),
    ("人民币", "banknote"), ("海边", "beach"), ("沙滩", "beach"), ("海滩", "beach"),
    ("雪山", "snowy mountain"), ("高山", "mountain"), ("森林", "forest"), ("树林", "forest"),
    ("湖泊", "lake"), ("湖边", "lake"), ("河流", "river"), ("瀑布", "waterfall"), ("日落", "sunset"),
    ("夕阳", "sunset"), ("日出", "sunrise"), ("晚上", "night"), ("夜里", "night"), ("夜晚", "night"),
    ("夜景", "night"), ("星空", "starry sky"), ("银河", "starry sky"), ("城市", "city skyline"),
    ("高楼", "city skyline"), ("街道", "city street"), ("街头", "city street"), ("马路", "city street"),
    ("公园", "park"), ("大桥", "bridge"), ("田野", "field"), ("农田", "field"), ("田地", "field"),
    ("花园", "garden"), ("泳池", "swimming pool"), ("寺庙", "temple"), ("教堂", "church"),
    ("酒店", "hotel room"), ("宾馆", "hotel room"), ("卧室", "bedroom"), ("厨房", "kitchen"),
    ("客厅", "living room"), ("教室", "classroom"), ("餐厅", "restaurant"), ("饭馆", "restaurant"),
    ("酒吧", "bar"), ("超市", "supermarket"), ("球场", "stadium"), ("机场", "airport"),
    ("车站", "train station"), ("飞机", "airplane"), ("航班", "airplane"), ("游轮", "boat"),
    ("轮船", "boat"), ("帆船", "boat"), ("火车", "train"), ("高铁", "train"), ("地铁", "train"),
    ("帐篷", "camping tent"), ("露营", "camping tent"), ("滑雪", "ski resort"), ("爬山", "hiking trail"),
    ("登山", "hiking trail"), ("徒步", "hiking trail"), ("下雨", "rainy"), ("雨天", "rainy"),
    ("雾天", "foggy"), ("晴天", "sunny"), ("阴天", "cloudy"), ("下雪", "snowing"), ("雪天", "snowing"),
    ("室内", "indoor"), ("室外", "outdoor"), ("户外", "outdoor"), ("自拍", "selfie"), ("写真", "portrait"),
    ("合照", "group photo"), ("合影", "group photo"), ("小孩", "child"), ("儿童", "child"), ("婴儿", "baby"),
    ("男人", "man"), ("女人", "woman"), ("情侣", "couple"), ("婚礼", "wedding"), ("结婚", "wedding"),
    ("婚宴", "wedding"), ("生日", "birthday party"), ("蛋糕", "cake"), ("毕业", "graduation"),
    ("人群", "crowd"), ("裙子", "skirt"), ("衬衫", "shirt"), ("西装", "suit"), ("制服", "uniform"),
    ("校服", "uniform"), ("泳装", "swimsuit"), ("泳衣", "swimsuit"), ("帽子", "hat"), ("墨镜", "sunglasses"),
    ("眼镜", "glasses"), ("围巾", "scarf"), ("婚纱", "wedding dress"), ("小狗", "puppy"), ("小猫", "kitten"),
    ("兔子", "rabbit"), ("蝴蝶", "butterfly"), ("熊猫", "panda"), ("大象", "elephant"), ("骆驼", "camel"),
    ("美食", "food"), ("甜品", "dessert"), ("甜点", "dessert"), ("水果", "fruit"), ("火锅", "hotpot"),
    ("烧烤", "barbecue"), ("面条", "noodles"), ("咖啡", "coffee"), ("奶茶", "milk tea"), ("红酒", "wine"),
    ("啤酒", "beer"), ("汽车", "car"), ("单车", "bicycle"), ("卡车", "truck"), ("电脑", "laptop"),
    ("电视", "television"), ("相机", "camera"), ("看书", "book"), ("报纸", "newspaper"), ("玩具", "toy"),
    ("气球", "balloon"), ("鲜花", "flower bouquet"), ("花束", "flower bouquet"), ("玫瑰", "roses"),
    ("行李", "luggage"), ("灯笼", "lantern"), ("烟花", "fireworks"), ("烟火", "fireworks"),
    ("文档", "document"), ("文件", "document"), ("发票", "invoice"), ("收据", "receipt"), ("门票", "ticket"),
    ("车票", "ticket"), ("机票", "ticket"), ("菜单", "menu"), ("海报", "poster"), ("白板", "whiteboard"),
    ("黑板", "blackboard"), ("截图", "screenshot"), ("护照", "passport"), ("纸币", "banknote"),
    ("硬币", "coin"), ("油画", "painting"), ("书法", "calligraphy"), ("地图", "map"), ("山", "mountain"),
    ("湖", "lake"), ("河", "river"), ("桥", "bridge"), ("船", "boat"), ("雾", "foggy"), ("狗", "dog"),
    ("猫", "cat"), ("鸟", "bird"), ("鱼", "fish"), ("马", "horse"), ("鹿", "deer"), ("书", "book"),
    ("花", "flower bouquet"), ("画", "painting"),
];

pub static ZH2CLIP: &[(&str, &str)] = &[
    ("红色连衣裙", "a red dress"), ("白色连衣裙", "a white dress"), ("蓝色连衣裙", "a blue dress"),
    ("红裙子", "a red dress"), ("连衣裙", "a dress"), ("向日葵", "sunflowers"), ("天安门", "Tiananmen Square"),
    ("摩天轮", "a ferris wheel"), ("放风筝", "flying a kite"), ("红裙", "a red dress"),
    ("白裙", "a white dress"), ("蓝裙", "a blue dress"), ("黄裙", "a yellow dress"),
    ("粉裙", "a pink dress"), ("旗袍", "a qipao cheongsam dress"),
    ("汉服", "traditional Chinese hanfu clothing"), ("和服", "a kimono"), ("礼服", "an evening gown"),
    ("雪人", "a snowman"), ("落叶", "autumn leaves"), ("樱花", "cherry blossoms"),
    ("长城", "the Great Wall of China"), ("故宫", "the Forbidden City"), ("灯塔", "a lighthouse"),
    ("温泉", "hot spring"), ("沙漠", "desert"), ("草原", "grassland prairie"), ("梯田", "terraced fields"),
    ("冲浪", "surfing"), ("潜水", "scuba diving"), ("跳伞", "skydiving"), ("骑车", "riding a bicycle"),
    ("骑马", "horse riding"), ("跑步", "running"), ("云海", "sea of clouds"), ("极光", "aurora"),
    ("彩虹", "a rainbow"), ("圣诞", "christmas"),
];

pub static RELATION: &[(&str, &str)] = &[
    ("自己", "我"), ("爸爸", "爸爸"), ("老爸", "爸爸"), ("我爸", "爸爸"), ("父亲", "爸爸"), ("妈妈", "妈妈"),
    ("老妈", "妈妈"), ("我妈", "妈妈"), ("母亲", "妈妈"), ("老婆", "老婆"), ("妻子", "老婆"), ("媳妇", "老婆"),
    ("老公", "老公"), ("丈夫", "老公"), ("儿子", "儿子"), ("女儿", "女儿"), ("宝宝", "宝宝"), ("孩子", "孩子"),
    ("爷爷", "爷爷"), ("奶奶", "奶奶"), ("外公", "外公"), ("姥爷", "外公"), ("外婆", "外婆"), ("姥姥", "外婆"),
    ("哥哥", "哥哥"), ("弟弟", "弟弟"), ("姐姐", "姐姐"), ("妹妹", "妹妹"), ("朋友", "朋友"), ("同学", "同学"),
    ("同事", "同事"), ("老师", "老师"), ("我", "我"), ("爸", "爸爸"), ("妈", "妈妈"),
];

pub static TAG_ZH_DISPLAY: &[(&str, &str)] = &[
    ("chat screenshot", "聊天截图"), ("snowy mountain", "雪山"), ("amusement park", "游乐场"),
    ("birthday party", "生日"), ("christmas tree", "圣诞树"), ("flower bouquet", "鲜花"),
    ("swimming pool", "游泳池"), ("train station", "高铁站"), ("wedding dress", "婚纱"),
    ("boarding pass", "登机牌"), ("city skyline", "城市"), ("camping tent", "露营"),
    ("hiking trail", "徒步"), ("city street", "马路"), ("living room", "客厅"), ("supermarket", "超市"),
    ("group photo", "合照"), ("down jacket", "羽绒服"), ("calligraphy", "书法"), ("starry sky", "星空"),
    ("hotel room", "宾馆"), ("restaurant", "饭馆"), ("ski resort", "滑雪"), ("graduation", "毕业典礼"),
    ("sunglasses", "太阳镜"), ("motorcycle", "摩托车"), ("television", "电视"), ("whiteboard", "白板"),
    ("blackboard", "黑板"), ("screenshot", "截图"), ("waterfall", "瀑布"), ("classroom", "教室"),
    ("butterfly", "蝴蝶"), ("newspaper", "报纸"), ("fireworks", "烟火"), ("mountain", "高山"),
    ("aquarium", "水族馆"), ("airplane", "航班"), ("portrait", "写真"), ("swimsuit", "泳衣"),
    ("elephant", "大象"), ("barbecue", "烧烤"), ("milk tea", "奶茶"), ("document", "文件"),
    ("passport", "护照"), ("banknote", "人民币"), ("painting", "油画"), ("sunrise", "日出"),
    ("bedroom", "卧室"), ("kitchen", "厨房"), ("concert", "音乐会"), ("stadium", "体育馆"),
    ("airport", "机场"), ("snowing", "雪天"), ("outdoor", "户外"), ("wedding", "婚宴"), ("uniform", "校服"),
    ("glasses", "眼镜"), ("dessert", "甜点"), ("noodles", "面条"), ("bicycle", "单车"), ("balloon", "气球"),
    ("luggage", "行李"), ("lantern", "灯笼"), ("invoice", "发票"), ("receipt", "收据"), ("id card", "身份证"),
    ("forest", "树林"), ("sunset", "夕阳"), ("bridge", "大桥"), ("garden", "花园"), ("temple", "寺庙"),
    ("church", "教堂"), ("museum", "博物馆"), ("office", "办公室"), ("cinema", "电影院"), ("cloudy", "阴天"),
    ("indoor", "室内"), ("selfie", "自拍"), ("family", "全家福"), ("couple", "情侣"), ("kitten", "小猫"),
    ("rabbit", "兔子"), ("hotpot", "火锅"), ("coffee", "咖啡"), ("laptop", "笔记本"), ("camera", "相机"),
    ("ticket", "机票"), ("poster", "海报"), ("beach", "海滩"), ("river", "河流"), ("night", "夜景"),
    ("field", "田地"), ("train", "地铁"), ("rainy", "雨天"), ("foggy", "雾天"), ("sunny", "晴天"),
    ("child", "儿童"), ("woman", "女人"), ("crowd", "很多人"), ("skirt", "裙子"), ("shirt", "衬衫"),
    ("scarf", "围巾"), ("puppy", "小狗"), ("horse", "马"), ("panda", "熊猫"), ("camel", "骆驼"),
    ("fruit", "水果"), ("truck", "卡车"), ("roses", "玫瑰"), ("lake", "湖边"), ("park", "公园"),
    ("cafe", "咖啡馆"), ("boat", "帆船"), ("baby", "婴儿"), ("cake", "蛋糕"), ("suit", "西装"), ("bird", "鸟"),
    ("fish", "鱼"), ("deer", "鹿"), ("food", "美食"), ("wine", "红酒"), ("beer", "啤酒"), ("book", "看书"),
    ("menu", "菜单"), ("coin", "硬币"), ("bar", "酒吧"), ("zoo", "动物园"), ("man", "男人"), ("hat", "鸭舌帽"),
    ("dog", "狗"), ("cat", "猫"), ("car", "汽车"), ("bus", "公交车"), ("toy", "玩具"), ("map", "地图"),
];

/// 触发定向 OCR 的文档类标签
pub static DOC_TAGS: &[&str] = &[
    "blackboard", "boarding pass", "chat screenshot", "document", "id card", "invoice", "map",
    "menu", "passport", "poster", "receipt", "screenshot", "ticket", "whiteboard",
];


#[cfg(test)]
mod tests {
    use super::*;

    /// 折行污染是最隐蔽的 bug：一个被折开的字符串字面量编译得过、运行时不报错，
    /// 但 ZH2TAG 里的多词标签永远匹配不上任何入库标签。这条测试是整个词表的地基。
    fn assert_clean(name: &str, items: &[&str]) {
        for s in items {
            assert!(!s.contains('\n'), "{name} 含换行: {s:?}");
            assert!(!s.contains('\r'), "{name} 含回车: {s:?}");
            assert_eq!(s.trim(), *s, "{name} 首尾有空白: {s:?}");
            assert!(!s.is_empty(), "{name} 有空串");
        }
    }

    fn assert_pairs_clean(name: &str, items: &[(&str, &str)]) {
        for (k, v) in items {
            assert!(!k.contains('\n'), "{name} key 含换行: {k:?}");
            assert!(!v.contains('\n'), "{name} value 含换行: {v:?}");
            assert!(!k.is_empty() && !v.is_empty(), "{name} 有空条目: {k:?}={v:?}");
        }
    }

    #[test]
    fn 词表无换行() {
        assert_clean("TAG_VOCAB", TAG_VOCAB);
        assert_clean("DOC_TAGS", DOC_TAGS);
        assert_pairs_clean("ZH2TAG", ZH2TAG);
        assert_pairs_clean("ZH2CLIP", ZH2CLIP);
        assert_pairs_clean("RELATION", RELATION);
        assert_pairs_clean("TAG_ZH_DISPLAY", TAG_ZH_DISPLAY);
    }

    #[test]
    fn 多词标签完整() {
        // 这些是最容易被折行破坏、且被 ZH2TAG 直接引用的值
        for want in ["city skyline", "swimming pool", "train station", "wedding dress",
                     "chat screenshot", "boarding pass", "snowy mountain", "birthday party"] {
            assert!(TAG_VOCAB.contains(&want), "TAG_VOCAB 缺 {want:?}");
        }
    }

    #[test]
    fn 中文映射指向存在的标签() {
        let tags: std::collections::HashSet<&str> = TAG_VOCAB.iter().copied().collect();
        for (zh, tag) in ZH2TAG {
            assert!(tags.contains(tag), "ZH2TAG[{zh:?}] = {tag:?} 不在 TAG_VOCAB 里");
        }
    }

    #[test]
    fn OCR标签都在闭集里() {
        let tags: std::collections::HashSet<&str> = TAG_VOCAB.iter().copied().collect();
        for t in DOC_TAGS {
            assert!(tags.contains(t), "DOC_TAGS 的 {t:?} 不在 TAG_VOCAB 里");
        }
    }

    #[test]
    fn 中文显示名覆盖全部标签() {
        // TAG_ZH_DISPLAY 由 ZH2TAG 反向生成，未被中文覆盖的标签应回落到英文
        let missing: Vec<&&str> = TAG_VOCAB
            .iter()
            .filter(|t| !TAG_ZH_DISPLAY.iter().any(|(k, _)| k == *t))
            .collect();
        assert!(
            missing.len() < TAG_VOCAB.len() / 2,
            "{} / {} 个标签没有中文名，回落逻辑可能没生效",
            missing.len(), TAG_VOCAB.len()
        );
    }
}
