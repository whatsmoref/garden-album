//! 最小 EXIF 解析器（只取需要的字段）
//!
//! 为什么自己写：`image` 0.25 已经没有 EXIF 解析（`kamadak-imaging` 从索引里消失了），
//! 而这恰恰是最容易出错的一块 —— 很多库只暴露 `get(36867)`，
//! 会静默拿不到 Exif IFD 里的 DateTimeOriginal（Python 版就踩过这个坑）。
//! 这里显式区分 IFD0 / Exif IFD(0x8769) / GPS IFD(0x8825)，并且
//! **DateTimeOriginal 优先于 0th IFD 的 DateTime**，取不到才退到文件 mtime。

use anyhow::{bail, Result};

#[derive(Debug, Clone, Default)]
pub struct ExifData {
    pub make: Option<String>,
    pub model: Option<String>,
    /// DateTimeOriginal（Exif IFD 36867）
    pub date_time_original: Option<String>,
    /// DateTimeDigitized（Exif IFD 36868）
    pub date_time_digitized: Option<String>,
    /// 0th IFD 306
    pub date_time: Option<String>,
    pub gps_lat: Option<f64>,
    pub gps_lon: Option<f64>,
    pub orientation: Option<u16>,
}

impl ExifData {
    /// 拍摄时间优先级：DateTimeOriginal > DateTimeDigitized > DateTime
    pub fn best_datetime(&self) -> Option<&str> {
        self.date_time_original
            .as_deref()
            .or(self.date_time_digitized.as_deref())
            .or(self.date_time.as_deref())
    }

    /// 有相机 Make + 拍摄时间才算实拍（扫描件/导出图通常只有 Model）
    pub fn from_camera(&self) -> bool {
        self.best_datetime().is_some() && self.make.as_deref().is_some_and(|m| !m.is_empty())
    }
}

const TAG_MAKE: u16 = 0x010F;
const TAG_MODEL: u16 = 0x0110;
const TAG_ORIENTATION: u16 = 0x0112;
const TAG_DATETIME: u16 = 0x0132;
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_GPS_IFD: u16 = 0x8825;
const GPS_LAT_REF: u16 = 1;
const GPS_LAT: u16 = 2;
const GPS_LON_REF: u16 = 3;
const GPS_LON: u16 = 4;
const EXIF_DATETIME_ORIGINAL: u16 = 0x9003;
const EXIF_DATETIME_DIGITIZED: u16 = 0x9004;

pub fn read(path: &std::path::Path) -> Result<ExifData> {
    let bytes = std::fs::read(path)?;
    let tiff = find_tiff_block(&bytes).ok_or_else(|| anyhow::anyhow!("无 EXIF"))?;
    parse_tiff(tiff)
}

/// 找出 TIFF 头的字节区间
fn find_tiff_block(b: &[u8]) -> Option<&[u8]> {
    // JPEG: APP1 段以 FF E1 开头，内容前 6 字节是 "Exif\0\0"
    if b.len() > 4 && b[0] == 0xFF && b[1] == 0xD8 {
        let mut i = 2usize;
        while i + 4 <= b.len() {
            if b[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = b[i + 1];
            // 填充字节
            if marker == 0xFF {
                i += 1;
                continue;
            }
            // 无长度字段的标记
            if (0xD0..=0xD9).contains(&marker) || marker == 0x01 || marker == 0xDA {
                break;
            }
            if i + 4 > b.len() {
                break;
            }
            let len = u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
            if len < 2 || i + 2 + len > b.len() {
                break;
            }
            let seg = &b[i + 4..i + 2 + len];
            if seg.len() >= 6 && &seg[0..6] == b"Exif\0\0" {
                return Some(&seg[6..]);
            }
            i += 2 + len;
        }
        return None;
    }
    // WebP: RIFF....WEBP 之后可能有 EXIF 块
    if b.len() > 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        let mut i = 12usize;
        while i + 8 <= b.len() {
            let fourcc = &b[i..i + 4];
            let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
            if fourcc == b"EXIF" {
                let start = i + 8;
                let end = (start + sz).min(b.len());
                let seg = &b[start..end];
                if seg.len() > 6 && &seg[0..6] == b"Exif\0\0" {
                    return Some(&seg[6..]);
                }
                if seg.len() > 4 {
                    return Some(seg); // 已经是裸 TIFF
                }
                return None;
            }
            i += 8 + sz + (sz & 1);
        }
        return None;
    }
    // PNG: eXIf chunk
    if b.len() > 8 && &b[1..4] == b"PNG" {
        let mut i = 8usize;
        while i + 8 <= b.len() {
            let sz = u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
            let typ = &b[i + 4..i + 8];
            if typ == b"eXIf" {
                let start = i + 8;
                let end = (start + sz).min(b.len());
                let seg = &b[start..end];
                return if seg.len() > 6 && &seg[0..6] == b"Exif\0\0" {
                    Some(&seg[6..])
                } else {
                    Some(seg)
                };
            }
            i += 12 + sz;
        }
        return None;
    }
    // TIFF 文件本身
    if b.len() > 8 && (&b[0..2] == b"II" || &b[0..2] == b"MM") {
        return Some(b);
    }
    None
}

#[derive(Clone)]
enum Val {
    Short(u16),
    Long(u32),
    Rationals(Vec<(u32, u32)>),
    Ascii(String),
    Undefined,
}

struct Reader<'a> {
    b: &'a [u8],
    le: bool,
}

impl<'a> Reader<'a> {
    fn u16(&self, off: usize) -> Result<u16> {
        if off + 2 > self.b.len() {
            bail!("越界");
        }
        Ok(if self.le {
            u16::from_le_bytes([self.b[off], self.b[off + 1]])
        } else {
            u16::from_be_bytes([self.b[off], self.b[off + 1]])
        })
    }
    fn u32(&self, off: usize) -> Result<u32> {
        if off + 4 > self.b.len() {
            bail!("越界");
        }
        Ok(if self.le {
            u32::from_le_bytes([self.b[off], self.b[off + 1], self.b[off + 2], self.b[off + 3]])
        } else {
            u32::from_be_bytes([self.b[off], self.b[off + 1], self.b[off + 2], self.b[off + 3]])
        })
    }
}

/// 解析一个 IFD，返回 (tag → 值, 下一个 IFD 偏移)
fn parse_ifd(r: &Reader, tiff: &[u8], ifd_off: usize) -> Result<(Vec<(u16, Val)>, Option<usize>)> {
    if ifd_off + 2 > tiff.len() {
        bail!("IFD 偏移越界");
    }
    let n = r.u16(ifd_off)? as usize;
    let mut out = Vec::with_capacity(n);
    let mut next = None;
    for i in 0..n {
        let e = ifd_off + 2 + i * 12;
        if e + 12 > tiff.len() {
            break;
        }
        let tag = r.u16(e)?;
        let typ = r.u16(e + 2)?;
        let count = r.u32(e + 4)? as usize;
        let size = match typ {
            1 | 2 => 1,          // BYTE / ASCII
            3 | 8 => 2,          // SHORT / SSHORT
            4 | 9 | 11 => 4,     // LONG / SLONG / FLOAT
            5 | 10 => 8,         // RATIONAL / SRATIONAL
            7 => 1,              // UNDEFINED
            _ => 0,
        };
        if size == 0 {
            continue;
        }
        let total = size * count;
        let data_off = if total <= 4 {
            e + 8
        } else {
            r.u32(e + 8)? as usize
        };
        if data_off + total > tiff.len() {
            continue;
        }
        let v = match typ {
            3 => Val::Short(r.u16(data_off)?),
            4 => Val::Long(r.u32(data_off)?),
            2 => {
                let raw = &tiff[data_off..data_off + total];
                let end = raw.iter().position(|c| *c == 0).unwrap_or(raw.len());
                Val::Ascii(String::from_utf8_lossy(&raw[..end]).trim_end().to_string())
            }
            5 => {
                let mut v = Vec::with_capacity(count);
                for k in 0..count {
                    let o = data_off + k * 8;
                    let num = r.u32(o)?;
                    let den = r.u32(o + 4)?;
                    v.push((num, den));
                }
                Val::Rationals(v)
            }
            _ => Val::Undefined,
        };
        out.push((tag, v));
    }
    if ifd_off + 2 + n * 12 + 4 <= tiff.len() {
        next = r.u32(ifd_off + 2 + n * 12).ok();
    }
    Ok((out, next))
}

fn ascii(v: &[Val]) -> Option<String> {
    match v.first() {
        Some(Val::Ascii(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}



fn rationals3(v: &[Val]) -> Option<f64> {
    match v.first() {
        Some(Val::Rationals(r)) if r.len() >= 3 => {
            let mut d = 0.0;
            for (n, den) in r.iter().take(3) {
                if *den == 0 {
                    return None;
                }
                d += n as f64 / den as f64;
            }
            // EXIF 存的是 [度, 分, 秒]
            let (dd, rest) = (d.trunc(), d - d.trunc());
            let min = (rest * 60.0).trunc();
            let sec = (rest * 60.0 - min) * 60.0;
            Some(dd + min / 60.0 + sec / 3600.0)
        }
        _ => None,
    }
}

pub fn parse_tiff(tiff: &[u8]) -> Result<ExifData> {
    if tiff.len() < 8 {
        bail!("TIFF 太短");
    }
    let le = match &tiff[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => bail!("字节序标记非法"),
    };
    let r = Reader { b: tiff, le };
    if r.u16(2)? != 42 {
        bail!("TIFF magic 不匹配");
    }
    let ifd0 = r.u32(4)? as usize;
    let (entries, next) = parse_ifd(&r, tiff, ifd0)?;

    let mut out = ExifData::default();
    let mut exif_off: Option<usize> = None;
    let mut gps_off: Option<usize> = None;
    for (tag, val) in &entries {
        match (*tag, val) {
            (TAG_MAKE, _) => out.make = ascii(std::slice::from_ref(val)),
            (TAG_MODEL, _) => out.model = ascii(std::slice::from_ref(val)),
            (TAG_DATETIME, _) => out.date_time = ascii(std::slice::from_ref(val)),
            (TAG_ORIENTATION, Val::Short(v)) => out.orientation = Some(*v),
            (TAG_EXIF_IFD, Val::Long(v)) => exif_off = Some(*v as usize),
            (TAG_GPS_IFD, Val::Long(v)) => gps_off = Some(*v as usize),
            _ => {}
        }
    }

    if let Some(off) = exif_off {
        if let Ok((ex, _)) = parse_ifd(&r, tiff, off) {
            for (tag, val) in &ex {
                match (*tag, val) {
                    (EXIF_DATETIME_ORIGINAL, _) => {
                        out.date_time_original = ascii(std::slice::from_ref(val))
                    }
                    (EXIF_DATETIME_DIGITIZED, _) => {
                        out.date_time_digitized = ascii(std::slice::from_ref(val))
                    }
                    _ => {}
                }
            }
        }
    }

    if let Some(off) = gps_off {
        if let Ok((g, _)) = parse_ifd(&r, tiff, off) {
            let mut lat = None;
            let mut lon = None;
            for (tag, val) in &g {
                match (*tag, val) {
                    (GPS_LAT, _) => lat = rationals3(std::slice::from_ref(val)),
                    (GPS_LON, _) => lon = rationals3(std::slice::from_ref(val)),
                    (GPS_LAT_REF, Val::Ascii(s)) if s.eq_ignore_ascii_case("S") && lat.is_some() => {
                        lat = lat.map(|v| -v)
                    }
                    (GPS_LON_REF, Val::Ascii(s)) if s.eq_ignore_ascii_case("W") && lon.is_some() => {
                        lon = lon.map(|v| -v)
                    }
                    _ => {}
                }
            }
            out.gps_lat = lat.map(round6);
            out.gps_lon = lon.map(round6);
        }
    }
    let _ = next;
    Ok(out)
}

fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

/// EXIF 的 "2024:07:13 09:54:00" → "2024-07-13 09:54:00"
pub fn normalize_dt(s: &str) -> String {
    let t = s.trim().trim_end_matches('\0').trim();
    if t.len() >= 19 && t.as_bytes()[4] == b':' && t.as_bytes()[7] == b':' {
        format!("{}-{}-{} {}", &t[0..4], &t[5..7], &t[8..10], &t[11..19])
    } else {
        t.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个最小 TIFF：IFD0(Make, DateTime, ExifIFD ptr) + ExifIFD(DateTimeOriginal)
    fn build_tiff(dt_original: &str, make: &str) -> Vec<u8> {
        let mut t = Vec::new();
        t.extend_from_slice(b"MM\x00\x2a");          // big-endian, magic 42
        t.extend_from_slice(&8u32.to_be_bytes());     // IFD0 @ 8
        // IFD0: 3 entries
        t.extend_from_slice(&3u16.to_be_bytes());
        // 271 Make (ASCII)
        let make_off = 8 + 2 + 3 * 12 + 4;
        t.extend_from_slice(&271u16.to_be_bytes());
        t.extend_from_slice(&2u16.to_be_bytes());     // ASCII
        t.extend_from_slice(&(make.len() as u32 + 1).to_be_bytes());
        t.extend_from_slice(&(make_off as u32).to_be_bytes());
        // 306 DateTime (ASCII)
        let dt = "2024:07:13 09:54:00";
        let dt_off = make_off + make.len() + 1;
        t.extend_from_slice(&306u16.to_be_bytes());
        t.extend_from_slice(&2u16.to_be_bytes());
        t.extend_from_slice(&20u32.to_be_bytes());
        t.extend_from_slice(&(dt_off as u32).to_be_bytes());
        // 0x8769 Exif IFD ptr
        let exif_off = dt_off + 20;
        t.extend_from_slice(&0x8769u16.to_be_bytes());
        t.extend_from_slice(&4u16.to_be_bytes());
        t.extend_from_slice(&1u32.to_be_bytes());
        t.extend_from_slice(&(exif_off as u32).to_be_bytes());
        t.extend_from_slice(&0u32.to_be_bytes());     // next IFD = 0
        // data 区
        t.extend_from_slice(make.as_bytes());
        t.push(0);
        t.extend_from_slice(dt.as_bytes());
        t.push(0);
        // Exif IFD: 1 entry (36867)
        t.extend_from_slice(&1u16.to_be_bytes());
        let dto_off = exif_off + 2 + 12 + 4;
        t.extend_from_slice(&0x9003u16.to_be_bytes());
        t.extend_from_slice(&2u16.to_be_bytes());
        t.extend_from_slice(&(dt_original.len() as u32 + 1).to_be_bytes());
        t.extend_from_slice(&(dto_off as u32).to_be_bytes());
        t.extend_from_slice(&0u32.to_be_bytes());
        t.extend_from_slice(dt_original.as_bytes());
        t.push(0);
        t
    }

    fn wrap_jpeg(tiff: &[u8]) -> Vec<u8> {
        let mut j = vec![0xFF, 0xD8];                       // SOI
        j.extend_from_slice(&[0xFF, 0xE1]);                 // APP1
        let len = (2 + 6 + tiff.len()) as u16;
        j.extend_from_slice(&len.to_be_bytes());
        j.extend_from_slice(b"Exif\0\0");
        j.extend_from_slice(tiff);
        j
    }

    #[test]
    fn 解析DateTimeOriginal而不是DateTime() {
        let tiff = build_tiff("2025:08:17 10:24:00", "Apple");
        let j = wrap_jpeg(&tiff);
        let block = find_tiff_block(&j).expect("应找到 TIFF");
        let e = parse_tiff(block).unwrap();
        // 关键：必须拿到 Exif IFD 的值，而不是 0th IFD 的 2024:07:13
        assert_eq!(e.best_datetime(), Some("2025:08:17 10:24:00"));
        assert_eq!(normalize_dt(e.best_datetime().unwrap()), "2025-08-17 10:24:00");
        assert_eq!(e.make.as_deref(), Some("Apple"));
        assert!(e.from_camera());
    }

    #[test]
    fn 缺DateTimeOriginal时退回DateTime() {
        let mut tiff = build_tiff("2025:08:17 10:24:00", "Apple");
        // 把 Exif IFD 的 tag 改成一个未知 tag，等于抹掉 DateTimeOriginal
        let idx = tiff
            .windows(2)
            .position(|w| w == 0x9003u16.to_be_bytes())
            .unwrap();
        tiff[idx] = 0x88;
        tiff[idx + 1] = 0x88;
        let j = wrap_jpeg(&tiff);
        let e = parse_tiff(find_tiff_block(&j).unwrap()).unwrap();
        assert_eq!(e.date_time_original, None);
        assert_eq!(e.best_datetime(), Some("2024:07:13 09:54:00"));
    }

    #[test]
    fn 无EXIF文件不报错() {
        assert!(find_tiff_block(&[0xFF, 0xD8, 0xFF, 0xD9]).is_none());
        assert!(find_tiff_block(b"not an image at all").is_none());
    }
}
