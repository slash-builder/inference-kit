//! Reading a model's `custom.license` string.
//!
//! The result is deliberately three-valued and **biased to refusal**: only a name that is clearly a well-known permissive licence is
//! [`NameClass::Permissive`]; a name that clearly restricts commercial use is [`NameClass::NonCommercial`]; everything else is
//! [`NameClass::Other`], which the registry treats as "not known to permit commercial use" (refused unless the catalog author sets
//! `commercial_use: "true"`).
//!
//! The order of the rules, which is what makes it fail closed:
//!
//! 1. Phrases that **affirm** commercial use ("commercial use permitted", "no commercial restrictions", "free for commercial use") are
//!    set aside, unless a negation sits just before them.
//! 2. What is left is searched for **negation** of commerce or profit within a few words (any language in the lexicon, plus CJK
//!    markers). A hit is `NonCommercial`, and nothing affirmative elsewhere in the string can undo it.
//! 3. A **scope word** (research, academic, evaluation, personal, educational, trial, demo, hobby) is `NonCommercial`, unless the string
//!    also affirmed commercial use, or the word only describes the release (`preview` or `fork` directly after it) and the rest is a
//!    recognised permissive licence.
//! 4. Otherwise every alternative (`OR`, `AND`, `/`, `;`, `|`) must be a recognised permissive licence to be `Permissive`.
//!
//! What is **not** classified as permissive, on purpose: the GPL and LGPL families and other copyleft licences, the OpenRAIL and
//! RAIL variants (`openrail++`, `CreativeML OpenRAIL-M`: they carry use restrictions), MPL and `CC-BY-SA`, and every custom or
//! community licence. They permit commercial use under conditions this crate cannot judge, so a catalog author vouches for them with
//! `commercial_use: "true"`.

/// What the licence name alone says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NameClass {
    /// Every alternative named is a recognised permissive licence.
    Permissive,
    /// The name restricts use to non-commercial, research, academic, evaluation or personal purposes, or negates commercial use.
    NonCommercial,
    /// Anything else, including an empty name.
    Other,
}

/// Lowercase words (letters and digits, Unicode-aware), with apostrophes removed (`don't` is `dont`) and `/ ; |` read as "or" so
/// `MIT/Apache-2.0` is two alternatives (a comma is just punctuation: `Apache License, Version 2.0`). `v2` is read as `2`.
fn words(licence: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (n, part) in licence.split(['/', ';', '|']).enumerate() {
        if n > 0 {
            out.push("or".to_string());
        }
        let part: String = part
            .to_lowercase()
            .chars()
            .filter(|c| !matches!(c, '\'' | '\u{2019}'))
            .collect();
        out.extend(
            part.split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(|w| match w.strip_prefix('v') {
                    Some(d) if !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()) => {
                        d.to_string()
                    }
                    _ => w.to_string(),
                }),
        );
    }
    out
}

/// Words that, near a word about commerce or profit, say the use is not allowed or is conditional. English, German, French, Spanish,
/// Italian, Portuguese. A word being here only matters next to a commerce word, so a false hit refuses a model and never admits one.
const RESTRICTIVE: &[&str] = &[
    // negators
    "non",
    "no",
    "not",
    "never",
    "without",
    "none",
    "cannot",
    "cant",
    "dont",
    "doesnt",
    "isnt",
    "arent",
    "wont",
    "mustnt",
    "shouldnt",
    "false",
    // prohibited, unauthorised, limited, conditional (English)
    "prohibited",
    "forbidden",
    "banned",
    "restricted",
    "disallowed",
    "denied",
    "excluded",
    "unauthorized",
    "unauthorised",
    "unlawful",
    "illegal",
    "limited",
    "permission",
    "requires",
    "required",
    "needs",
    "need",
    // German
    "nicht",
    "kein",
    "keine",
    "keinen",
    "keiner",
    "ohne",
    "verboten",
    "untersagt",
    // French
    "sans",
    "pas",
    "interdit",
    "interdite",
    "interdits",
    // Spanish, Portuguese
    "sin",
    "prohibido",
    "prohibida",
    "prohibidos",
    "proibido",
    "proibida",
    // Italian
    "senza",
    "vietato",
    "vietata",
    "vietati",
];

/// CJK phrasings, matched as substrings of the lowercased name because those scripts have no spaces between words.
const CJK_NON_COMMERCIAL: &[&str] = &[
    "非商用",
    "禁止商用",
    "商用禁止",
    "商用不可",
    "不可商用",
    "不得商用",
    "禁商用",
    "非商業",
    "非商业",
    "禁止商業",
    "禁止商业",
    "非営利",
    "비상업",
    "상업적 이용 금지",
    "상업적 사용 금지",
    "상업적 이용 불가",
];

/// How many words either side of "commercial" or "profit" a [`RESTRICTIVE`] word counts: "not for commercial use" (2),
/// "cannot be used for commercial purposes" (4). Wide on purpose: a false hit only refuses a model.
const WINDOW: usize = 4;

/// A word about commerce: commercial, commercially, commerciale, kommerziell, comercial.
fn is_commerce_root(w: &str) -> bool {
    w.starts_with("commerc") || w.starts_with("kommerz") || w.starts_with("comerc")
}

/// A word about commerce or profit.
fn is_commerce_word(w: &str) -> bool {
    is_commerce_root(w) || w == "profit" || w == "profits"
}

/// Words that limit a licence to a kind of use other than commercial.
fn is_scope_word(w: &str) -> bool {
    matches!(
        w,
        "research"
            | "academic"
            | "evaluation"
            | "personal"
            | "educational"
            | "hobby"
            | "trial"
            | "demo"
    )
}

/// A scope word directly followed by one of these only describes the release ("research preview", "personal fork"). Deliberately just
/// these two: words such as `version`, `edition`, `copy`, `build`, `release` or `model` name what the terms apply to ("evaluation
/// version"), not that the terms are unaffected.
fn is_release_word(w: &str) -> bool {
    matches!(w, "preview" | "fork")
}

/// After a commerce word (and a few connecting words) these say commercial use is fine.
fn is_affirmative(w: &str) -> bool {
    matches!(
        w,
        "permitted"
            | "allowed"
            | "ok"
            | "okay"
            | "permissible"
            | "unrestricted"
            | "granted"
            | "welcome"
            | "yes"
            | "true"
            | "fine"
            | "free"
            | "unlimited"
    )
}

/// Connecting words that may sit between "commercial" and "permitted".
fn is_connector(w: &str) -> bool {
    matches!(
        w,
        "use"
            | "usage"
            | "uses"
            | "purposes"
            | "purpose"
            | "is"
            | "are"
            | "be"
            | "may"
            | "can"
            | "also"
            | "fully"
    )
}

fn is_use_word(w: &str) -> bool {
    matches!(w, "use" | "usage" | "uses" | "purposes" | "purpose")
}

fn is_restriction_noun(w: &str) -> bool {
    matches!(
        w,
        "restriction"
            | "restrictions"
            | "limitation"
            | "limitations"
            | "limits"
            | "limit"
            | "strings"
    )
}

/// If the words at `i` open an affirmation of commercial use, the number of words it covers.
///
/// - `no commercial restrictions`, `no restrictions on commercial use`
/// - `commercial use permitted`, `commercial use is allowed`, `commercial=true`, `commercial use OK`
/// - `free for commercial use`
fn affirmation_at(w: &[String], i: usize) -> Option<usize> {
    let at = |k: usize| w.get(i + k).map(String::as_str);
    // no commercial restrictions
    if at(0) == Some("no")
        && at(1).is_some_and(is_commerce_root)
        && at(2).is_some_and(is_restriction_noun)
    {
        return Some(3);
    }
    // no restrictions on commercial use
    if at(0) == Some("no")
        && at(1).is_some_and(is_restriction_noun)
        && matches!(at(2), Some("on" | "for" | "to" | "of" | "regarding"))
        && at(3).is_some_and(is_commerce_root)
    {
        return Some(if at(4).is_some_and(is_use_word) { 5 } else { 4 });
    }
    // commercial use (is) permitted
    if at(0).is_some_and(is_commerce_root) {
        let mut j = 1;
        while at(j).is_some_and(is_connector) {
            j += 1;
        }
        if at(j).is_some_and(is_affirmative) {
            return Some(j + 1);
        }
    }
    // free for commercial use
    if at(0) == Some("free") {
        let j = if at(1) == Some("for") { 2 } else { 1 };
        if at(j).is_some_and(is_commerce_root) {
            return Some(j + 1 + usize::from(at(j + 1).is_some_and(is_use_word)));
        }
    }
    None
}

/// Words that say nothing about the terms and are dropped before a licence id is looked for: `no attribution required`, or a bare
/// `no restrictions` at the end.
fn noise_at(w: &[String], i: usize) -> Option<usize> {
    let at = |k: usize| w.get(i + k).map(String::as_str);
    if at(0) == Some("no") && at(1) == Some("attribution") && at(2) == Some("required") {
        return Some(3);
    }
    if at(0) == Some("attribution") && at(1) == Some("required") {
        return Some(2);
    }
    if at(0) == Some("no")
        && at(1).is_some_and(is_restriction_noun)
        && !matches!(at(2), Some("on" | "for" | "to" | "of" | "in" | "regarding"))
    {
        return Some(2);
    }
    None
}

/// The words with affirmations of commercial use (and noise) removed, and whether any affirmation was found.
///
/// An affirmation does not count when a restrictive word sits in the [`WINDOW`] words just before it ("no commercial use permitted",
/// "not for commercial use allowed"): that is a negation, and it stays in for the next rule to find.
fn set_aside_affirmations(w: &[String]) -> (Vec<String>, bool) {
    let mut out: Vec<String> = Vec::new();
    let mut affirmed = false;
    let mut i = 0;
    while i < w.len() {
        if let Some(n) = noise_at(w, i) {
            i += n;
            continue;
        }
        if let Some(n) = affirmation_at(w, i) {
            let starts_with_no = w[i] == "no";
            let before = &out[out.len().saturating_sub(WINDOW)..];
            let negated_before =
                !starts_with_no && before.iter().any(|x| RESTRICTIVE.contains(&x.as_str()));
            if !negated_before {
                affirmed = true;
                i += n;
                continue;
            }
        }
        out.push(w[i].clone());
        i += 1;
    }
    (out, affirmed)
}

/// The name negates commerce or profit: `non-commercial`, `not for commercial use`, `commercial use prohibited`, `non-profit`, the
/// Creative Commons `NC` family, `commercial=false`, and the same in the other languages of [`RESTRICTIVE`].
fn negates_commercial_use(w: &[String]) -> bool {
    let has = |x: &str| w.iter().any(|y| y == x);
    if has("noncommercial") || has("nonprofit") || has("nc") {
        // `nc` is the Creative Commons NC family (cc-by-nc-4.0, cc-by-nc-sa, cc-by-nc-nd), matched as a whole word
        return true;
    }
    for (i, word) in w.iter().enumerate() {
        if !is_commerce_word(word) {
            continue;
        }
        let lo = i.saturating_sub(WINDOW);
        let hi = (i + WINDOW + 1).min(w.len());
        if w[lo..hi]
            .iter()
            .enumerate()
            .any(|(k, x)| lo + k != i && RESTRICTIVE.contains(&x.as_str()))
        {
            return true;
        }
    }
    false
}

/// Scope words that are not just describing a release. Returns `(restricting, any)`.
fn scope_words(w: &[String]) -> (bool, bool) {
    let mut restricting = false;
    let mut any = false;
    for (i, word) in w.iter().enumerate() {
        if !is_scope_word(word) {
            continue;
        }
        any = true;
        if !w.get(i + 1).is_some_and(|n| is_release_word(n)) {
            restricting = true;
        }
    }
    (restricting, any)
}

/// Words dropped before matching a licence id: `MIT License`, `Apache License, Version 2.0`, `The Unlicense`, `Licensed under MIT`.
fn is_filler(w: &str) -> bool {
    matches!(
        w,
        "the"
            | "license"
            | "licence"
            | "licenses"
            | "licensed"
            | "licenced"
            | "version"
            | "software"
            | "international"
            | "unported"
            | "generic"
            | "universal"
            | "under"
            | "released"
            | "distributed"
            | "available"
    )
}

/// Ids of licences that permit commercial use without further conditions this crate would need to judge, as hyphen-joined words.
const PERMISSIVE: &[&str] = &[
    "mit",
    "mit-0",
    "apache",
    "apache-2",
    "apache-2-0",
    "apache2",
    "apache2-0",
    "bsd",
    "bsd-2",
    "bsd-3",
    "bsd-2-clause",
    "bsd-3-clause",
    "0bsd",
    "new-bsd",
    "modified-bsd",
    "simplified-bsd",
    "revised-bsd",
    "isc",
    "cc0",
    "cc0-1-0",
    "unlicense",
    "zlib",
    "cc-by",
    "cc-by-3-0",
    "cc-by-4-0",
];

/// Whether one alternative (words between `or` / `and`) is made of recognised permissive ids, optionally with release words
/// ("research preview", "personal fork") in between.
fn is_permissive_alternative(alt: &[String]) -> bool {
    // "Creative Commons Attribution 4.0" is "cc by 4.0"
    let mut a: Vec<String> = Vec::new();
    let mut i = 0;
    while i < alt.len() {
        if alt[i..].len() >= 3 && alt[i..i + 3] == ["creative", "commons", "attribution"] {
            a.push("cc".into());
            a.push("by".into());
            i += 3;
        } else {
            a.push(alt[i].clone());
            i += 1;
        }
    }
    let a: Vec<&str> = a
        .iter()
        .map(String::as_str)
        .filter(|w| !is_filler(w))
        .collect();
    // Walk the words: take the longest run that is a known id, skip release words, fail on anything else. More than one id in a row is
    // fine ("MIT, Apache-2.0" has no "or" in it once the comma is dropped); at least one id is required.
    let (mut pos, mut ids) = (0, 0);
    while pos < a.len() {
        if let Some(n) = (1..=a.len() - pos)
            .rev()
            .find(|n| PERMISSIVE.contains(&a[pos..pos + n].join("-").as_str()))
        {
            pos += n;
            ids += 1;
        } else if is_release_word(a[pos]) || is_scope_word(a[pos]) {
            pos += 1;
        } else {
            return false;
        }
    }
    ids > 0
}

/// Read a licence name. See the module documentation for the rules and what is deliberately left as [`NameClass::Other`].
pub(crate) fn classify(licence: &str) -> NameClass {
    let lower = licence.to_lowercase();
    if CJK_NON_COMMERCIAL.iter().any(|m| lower.contains(m)) {
        return NameClass::NonCommercial;
    }
    let mut w = words(licence);
    // "Apache-2.0-or-later" is Apache-2.0: drop "or later" / "and later" so the `or` is not read as an alternative
    let mut i = 0;
    while i + 1 < w.len() {
        if matches!(w[i].as_str(), "or" | "and") && w[i + 1] == "later" {
            w.drain(i..i + 2);
        } else {
            i += 1;
        }
    }
    let (w, affirmed) = set_aside_affirmations(&w);
    if negates_commercial_use(&w) {
        return NameClass::NonCommercial;
    }
    let (restricting_scope, any_scope) = scope_words(&w);
    if restricting_scope && !affirmed {
        return NameClass::NonCommercial;
    }
    let alternatives: Vec<&[String]> = w
        .split(|x| x == "or" || x == "and")
        .filter(|a| !a.is_empty())
        .collect();
    if !alternatives.is_empty() && alternatives.iter().all(|a| is_permissive_alternative(a)) {
        return NameClass::Permissive;
    }
    if any_scope && !affirmed {
        // "research preview" with no permissive licence behind it: still a research release, so not commercial
        return NameClass::NonCommercial;
    }
    NameClass::Other
}

#[cfg(test)]
mod tests {
    use super::*;
    use NameClass::{NonCommercial, Other, Permissive};

    fn all(class: NameClass, names: &[&str]) {
        for l in names {
            assert_eq!(classify(l), class, "{l}");
        }
    }

    #[test]
    fn non_commercial_phrasings_in_several_forms() {
        all(
            NonCommercial,
            &[
                "non-commercial research",
                "Non-Commercial",
                "noncommercial",
                "NonCommercial",
                "non commercial",
                "No commercial use",
                "Not commercial",
                "Not for commercial use",
                "not-for-commercial-use",
                "For non commercial use only",
                "Commercial use prohibited",
                "Commercial use is not permitted",
                "Commercial use forbidden",
                "Commercial usage disallowed",
                "Free for non-profit use",
                "nonprofit use only",
                "Not for profit",
                "Nicht kommerziell",
                "Nicht-kommerzielle Nutzung",
                "Keine kommerzielle Nutzung",
                "Usage non commercial",
                "Uso no comercial",
                "Uso non commerciale",
                "research-only",
                "Research use only",
                "Academic use only",
                "evaluation only",
                "Free for personal and research use",
                "For personal use",
                "Educational use only",
                "CC-BY-NC-4.0",
                "cc-by-nc-sa-4.0",
                "CC BY-NC-ND 4.0",
                "cc_by_nc_4.0",
                "FLUX.1-dev Non-Commercial License",
                "Stability AI Non Commercial Research Community License",
                "Apache-2.0 research use only",
                "MIT OR CC-BY-NC-4.0",
                "research preview",
            ],
        );
    }

    #[test]
    fn negations_in_more_wordings_and_languages() {
        all(
            NonCommercial,
            &[
                "don't use commercially",
                "Don\u{2019}t use commercially",
                "do not use commercially",
                "cannot be used for commercial purposes",
                "can't be used commercially",
                "may not be used commercially",
                "Commercial use unauthorized",
                "Commercial use unauthorised",
                "commercial use unlawful",
                "commercial use is illegal",
                "commercial use limited",
                "Commercial use needs permission",
                "commercial use requires a licence",
                "commercial use requires a separate license",
                "Uso comercial prohibido",
                "Uso comercial proibido",
                "Usage commercial interdit",
                "Sans usage commercial",
                "Nutzung kommerziell verboten",
                "Kommerzielle Nutzung untersagt",
                "Vietato uso commerciale",
                "Senza uso commerciale",
                "MIT; commercial=false",
                "MIT; commercial_use: false",
                "Apache-2.0, commercial: no",
                "商用禁止",
                "禁止商用",
                "非商用",
                "非商业用途",
                "비상업적 용도",
                "Apache-2.0 (非商用)",
            ],
        );
    }

    #[test]
    fn a_negation_stays_a_negation_even_beside_an_affirmation() {
        all(
            NonCommercial,
            &[
                "not for commercial use, no restrictions on research",
                "Not for commercial use. Commercial use OK for partners",
                "no commercial use permitted",
                "not for commercial use allowed",
                "commercial use permitted; non-commercial only",
                "MIT, commercial use permitted, cc-by-nc",
                "no commercial restrictions, but research only and not for commercial use",
            ],
        );
    }

    #[test]
    fn a_negation_is_found_within_the_window_and_not_beyond_it() {
        // three unrelated words between the negator and the commerce word (4 apart): seen; four (5 apart): not. Literal numbers, so
        // changing WINDOW has to change this test too.
        let near = format!("no {} commercial", ["x"; 3].join(" "));
        let far = format!("no {} commercial", ["x"; 4].join(" "));
        assert_eq!(classify(&near), NonCommercial, "{near}");
        assert_eq!(classify(&far), Other, "{far}");
        // and the other side of the commerce word
        let near = format!("commercial {} prohibited", ["x"; 3].join(" "));
        let far = format!("commercial {} prohibited", ["x"; 4].join(" "));
        assert_eq!(classify(&near), NonCommercial, "{near}");
        assert_eq!(classify(&far), Other, "{far}");
    }

    #[test]
    fn affirmations_of_commercial_use_are_not_read_as_restrictions() {
        all(
            Permissive,
            &[
                "MIT, no commercial restrictions",
                "MIT, no restrictions on commercial use",
                "Apache 2.0 - commercial use permitted, no restrictions",
                "Apache-2.0 (commercial use OK, no attribution required)",
                "MIT (free for commercial use)",
                "Apache-2.0; commercial=true",
                "Apache-2.0, commercial use allowed",
                "MIT - commercial use is permitted",
                "MIT, commercial usage unrestricted",
                "MIT, no commercial limitations",
            ],
        );
        // not a licence on their own: the id is what makes it permissive, and "commercial_use" is the catalog's field
        all(
            Other,
            &[
                "commercial use permitted",
                "free for commercial use",
                "no commercial restrictions",
            ],
        );
        // explicit commercial wording outweighs a scope word, but not a negation
        assert_eq!(
            classify("Apache-2.0 for research and commercial use permitted"),
            Other
        );
        assert_eq!(
            classify("MIT research, commercial use permitted"),
            Permissive
        );
    }

    #[test]
    fn well_known_permissive_licences_in_common_spellings() {
        all(
            Permissive,
            &[
                "MIT",
                "mit",
                "MIT License",
                "The MIT License",
                "MIT-0",
                "Licensed under MIT",
                "Released under the MIT license",
                "Apache-2.0",
                "apache 2.0",
                "Apache License 2.0",
                "Apache License, Version 2.0",
                "Apache License v2.0",
                "Apache Software License 2.0",
                "apache2",
                "Apache-2.0-or-later",
                "BSD",
                "BSD-2-Clause",
                "BSD 2-Clause License",
                "BSD-3-Clause",
                "BSD 3-Clause License",
                "0BSD",
                "ISC",
                "ISC License",
                "CC0-1.0",
                "CC0 1.0 Universal",
                "Unlicense",
                "The Unlicense",
                "Zlib",
                "CC-BY",
                "CC BY",
                "CC-BY-4.0",
                "CC BY 4.0",
                "cc-by-3.0",
                "Creative Commons Attribution 4.0 International",
                "Apache-2.0 OR MIT",
                "Apache-2.0 or MIT",
                "MIT/Apache-2.0",
                "MIT, Apache-2.0",
                "MIT AND BSD-3-Clause",
                "(MIT OR Apache-2.0)",
            ],
        );
    }

    #[test]
    fn an_alternative_must_itself_be_a_licence() {
        // "/" and "or" separate alternatives; a release word alone is not one
        assert_eq!(classify("MIT/fork"), Other);
        assert_eq!(classify("MIT or fork"), Other);
        assert_eq!(classify("MIT fork"), Permissive);
        assert_eq!(classify("MIT/Apache-2.0"), Permissive);
    }

    #[test]
    fn copyleft_rail_and_community_licences_are_left_unrecognised_on_purpose() {
        all(
            Other,
            &[
                "GPL-3.0",
                "GPL-3.0-or-later",
                "GNU General Public License v3",
                "LGPL-2.1",
                "AGPL-3.0",
                "MPL-2.0",
                "CC-BY-SA-4.0",
                "openrail++",
                "CreativeML OpenRAIL-M",
                "OpenRAIL-M",
                "bigscience-bloom-rail-1.0",
                "llama3 community license",
                "Gemma Terms of Use",
                "proprietary",
                "other",
                "custom licence v3",
                "see LICENSE file",
                "MIT AND GPL-3.0",
                "Apache-2.0 WITH LLVM-exception",
                "",
                "   ",
            ],
        );
    }

    #[test]
    fn only_a_preview_or_fork_marker_leaves_a_permissive_licence_alone() {
        // the licence id decides; "research preview" and "personal fork" say what the release is, not what the terms allow
        all(
            Permissive,
            &[
                "Apache-2.0 (research preview)",
                "MIT personal fork",
                "MIT (evaluation preview)",
                "Apache-2.0 (educational fork)",
            ],
        );
        // any other scope word, however it reads, restricts
        all(
            NonCommercial,
            &[
                "MIT research",
                "MIT (academic)",
                "Apache-2.0 for personal use",
                "MIT evaluation version",
                "MIT evaluation copy",
                "MIT evaluation build",
                "Apache-2.0 evaluation release",
                "MIT personal edition",
                "MIT educational copy",
                "MIT (research model)",
                "Apache-2.0 research version",
                "MIT trial",
                "MIT demo",
                "MIT hobby",
                "custom research preview",
            ],
        );
    }

    #[test]
    fn nc_is_a_word_not_letters_inside_another_word() {
        assert_eq!(classify("Apache-2.0 (Lancaster fork)"), Other);
        assert_eq!(classify("Funcsoft licence"), Other);
    }
}
