# font_theme_helpers.R
# helper to register Latin Modern from TeX and set ggplot theme
# also handles various other formatting tasks

init_lmodern_theme <- function(base_size = 10,
                               family_name = "LMRoman",
                               set_theme = TRUE,
                               enable_showtext = TRUE) {
  if (!requireNamespace("sysfonts", quietly = TRUE) ||
      !requireNamespace("showtext", quietly = TRUE)) {
    stop("Please install.packages(c('sysfonts','showtext'))")
  }
  
  # Try to locate LM OTF/TTF via kpsewhich or common TeX roots
  kpse_find <- function(fname) {
    p <- tryCatch(system2("kpsewhich", fname, stdout = TRUE),
                  error = function(e) character())
    if (length(p) && nzchar(p[1])) normalizePath(p[1], winslash = "/") else ""
  }
  
  find_paths <- function() {
    files <- list(
      regular    = c(kpse_find("lmroman10-regular.otf"),    kpse_find("lmroman10-regular.ttf")),
      bold       = c(kpse_find("lmroman10-bold.otf"),       kpse_find("lmroman10-bold.ttf")),
      italic     = c(kpse_find("lmroman10-italic.otf"),     kpse_find("lmroman10-italic.ttf")),
      bolditalic = c(kpse_find("lmroman10-bolditalic.otf"), kpse_find("lmroman10-bolditalic.ttf"))
    )
    paths <- vapply(files, function(x) x[which(nzchar(x))[1]], character(1))
    
    if (any(!nzchar(paths))) {
      roots <- c("C:/texlive", "C:/Program Files/MiKTeX",
                 file.path(Sys.getenv("LOCALAPPDATA"), "Programs", "MiKTeX"))
      find1 <- function(pat) {
        hits <- unlist(lapply(roots, function(r)
          list.files(r, pattern = pat, recursive = TRUE, full.names = TRUE, ignore.case = TRUE)),
          use.names = FALSE)
        if (length(hits)) normalizePath(hits[1], winslash = "/") else ""
      }
      if (!nzchar(paths["regular"]))    paths["regular"]    <- find1("lmroman10-regular\\.(otf|ttf)$")
      if (!nzchar(paths["bold"]))       paths["bold"]       <- find1("lmroman10-bold\\.(otf|ttf)$")
      if (!nzchar(paths["italic"]))     paths["italic"]     <- find1("lmroman10-italic\\.(otf|ttf)$")
      if (!nzchar(paths["bolditalic"])) paths["bolditalic"] <- find1("lmroman10-bolditalic\\.(otf|ttf)$")
    }
    paths
  }
  
  paths <- find_paths()
  if (any(!nzchar(paths))) stop("Couldn't locate Latin Modern OTF/TTF in your TeX install.")
  
  # Register only once
  if (!(family_name %in% sysfonts::font_families())) {
    sysfonts::font_add(family_name,
                       regular    = paths["regular"],
                       bold       = paths["bold"],
                       italic     = paths["italic"],
                       bolditalic = paths["bolditalic"]
    )
  }
  
  if (enable_showtext) showtext::showtext_auto()
  
  if (set_theme) {
    ggplot2::theme_set(ggplot2::theme_bw(base_size = base_size, base_family = family_name))
  }
  
  invisible(paths)
}

# ---------- Tiny formatters ----------
fmt_pval <- function(p, digits = 3, eps = 1e-4) {
  ifelse(
    is.na(p), NA_character_,
    ifelse(p < eps,
           paste0("<", sprintf(paste0("%.", digits, "f"), eps)),
           sprintf(paste0("%.", digits, "f"), p))
  )
}

# Scientific notation for LaTeX: returns "$m \times 10^{e}$", handles 0/NA cleanly
fmt_sci_tex <- function(x, digits = 2) {
  out <- rep("--", length(x))
  # NAs, ±Inf -> "--"
  ok  <- is.finite(x)
  if (any(ok)) {
    z  <- x[ok]
    zero <- z == 0
    if (any(zero)) out[which(ok)[zero]] <- "$0$"
    nz <- !zero
    if (any(nz)) {
      zz  <- z[nz]
      ex  <- floor(log10(abs(zz)))
      man <- zz / (10 ^ ex)
      out[which(ok)[nz]] <- sprintf("$%s\\times 10^{%d}$",
                                    formatC(man, format = "f", digits = digits),
                                    ex)
    }
  }
  out
}


fmt_num  <- function(x, digits = 3) {
  out <- ifelse(is.finite(x), formatC(x, format="f", digits=digits),
                NA_character_); out[is.na(out)] <- "--"; out
                }
fmt_pct  <- function(x, digits = 1, scale = 100) {
  out <- ifelse(is.finite(x), sprintf(paste0("%.", digits, "f"),
                                      x * scale), NA_character_);
  out[is.na(out)] <- "--"; out
  }
fmt_date <- function(x) {
  if (inherits(x, "Date")) y <- format(x, "%Y-%m") else {
    xx <- suppressWarnings(as.Date(x)); y <- ifelse(is.na(xx),
                                                    NA_character_, format(xx, "%Y-%m"))
  }
  y[is.na(y)] <- "--"; y
}

# ---------- Grouping on pretty model names ----------
# Vectorised grouping on pretty model names
model_group_from_pretty <- function(model_pretty) {
  m <- tolower(as.character(model_pretty))
  dplyr::case_when(
    grepl("na[iï]ve|rolling mean|pnc|auto-?arima|\\barima\\b", m, perl = TRUE) ~ "Baseline Models",
    grepl("\\bbvar\\b", m, perl = TRUE)                                          ~ "BVAR Models",
    grepl("dynamic factor|\\bdfm\\b|dist\\.? ?random forest|\\bdrf\\b", m, perl = TRUE) ~ "Multivariate \\& ML Models",
    TRUE                                                                          ~ "Other Models"
  )
}

# ---------- Collect/merge: KS + boundary + e-crossings (from the *_RESULTS_ALL tibbles) ----------
collect_region_calibration <- function(region_key, horizons,
                                       ks_all, boundary_all, crossing_all) {
  ks1 <- ks_all %>%
    dplyr::filter(region == region_key, horizon %in% horizons) %>%
    dplyr::transmute(model, horizon = as.integer(horizon), ks_p = p_value)  # your KS uses p_value.
  
  bd1 <- boundary_all %>%
    dplyr::filter(region == region_key, horizon %in% horizons) %>%
    dplyr::transmute(model, horizon = as.integer(horizon), boundary_share)
  
  # Prefer a single method per (model,h). If multiple, keep the row with largest e_max; fall back to first.
  cr1 <- crossing_all %>%
    dplyr::filter(region == region_key, horizon %in% horizons) %>%
    dplyr::group_by(model, horizon) %>%
    dplyr::arrange(dplyr::desc(ifelse(is.finite(e_max), e_max, -Inf)), .by_group = TRUE) %>%
    dplyr::slice(1) %>%
    dplyr::ungroup() %>%
    dplyr::transmute(model, horizon = as.integer(horizon),
                     first_cross_10, first_cross_100, e_max, e_last)
  
  out <- ks1 %>%
    dplyr::full_join(bd1, by = c("model","horizon")) %>%
    dplyr::full_join(cr1, by = c("model","horizon")) %>%
    dplyr::mutate(model_group = model_group_from_pretty(model)) %>%
    dplyr::arrange(factor(model_group,
                          levels = c("Baseline Models","BVAR Models","Multivariate \\& ML Models","Other Models")),
                   model, horizon)
  out
}

# ---------- LaTeX block for a set of horizons (compact, booktabs-ready) ----------
latex_calibration_block <- function(df_region, hset,
                                    digits_p = 3, digits_pct = 1, digits_e = 2) {
  subcols <- c(
    "\\makecell{\\textbf{KS Test} \\\\ \\textbf{($\\mathbf{p}$)} }",
    "\\makecell{\\textbf{Edge} \\\\ \\textbf{freq. (\\%)} }",
    "\\textbf{max $\\mathbf{e}$}"
  )
  n_sub   <- length(subcols)
  n_cols  <- 1 + length(hset) * n_sub
  colspec <- paste0("l ", paste(rep("c", n_cols - 1), collapse = " "))
  
  header1 <- paste0(" & ",
                    paste(sprintf("\\multicolumn{%d}{c}{\\textbf{h=%s}}", n_sub, hset), collapse = " & "),
                    " \\\\")
  cmid <- paste(vapply(seq_along(hset), function(i) {
    l <- 1 + (i - 1) * n_sub + 1; r <- l + n_sub - 1; sprintf("\\cmidrule(lr){%d-%d}", l, r)
  }, character(1)), collapse = " ")
  header2 <- paste0("\\textbf{Model} & ", paste(rep(subcols, times = length(hset)), collapse = " & "), " \\\\")
  
  dd <- df_region %>% dplyr::filter(horizon %in% hset)
  models <- dd %>% dplyr::distinct(model, model_group) %>%
    dplyr::arrange(factor(model_group,
                          levels = c("Baseline Models","BVAR Models","Multivariate \\& ML Models","Other Models")), model)
  
  row_for_model <- function(m) {
    vals <- c()
    for (h in hset) {
      row <- dd %>% dplyr::filter(model == m, horizon == h)
      if (nrow(row) == 0) vals <- c(vals, rep("--", n_sub)) else {
        vals <- c(vals,
                  fmt_num(row$ks_p, digits_p),
                  fmt_pct(row$boundary_share, digits_pct),
                  fmt_sci_tex(row$e_max, digits_e))
      }
    }
    paste0(m, " & ", paste(vals, collapse = " & "), " \\\\")
  }
  
  body <- c()
  groups <- unique(models$model_group)
  for (i in seq_along(groups)) {
    grp <- groups[i]
    grp_models <- models %>% dplyr::filter(model_group == grp)
    
    if (i > 1) {
      body <- c(body, "\\midrule")
    }
    
    body <- c(
      body,
      sprintf("\\multicolumn{%d}{l}{\\textit{%s}} \\\\", n_cols, grp),
      vapply(grp_models$model, row_for_model, character(1))
    )
  }
  
  paste0(
    "\\begin{tabular}{", colspec, "}\n\\toprule\n",
    header1, "\n", cmid, "\n", header2, "\n\\midrule\n",
    paste(body, collapse = "\n"), "\n\\bottomrule\n\\end{tabular}"
  )
}

# ---------- One-shot table wrapper (region, blocks) ----------
render_region_calibration_tables_from_all <- function(region_key,
                                                      horizon_blocks = list(c(1,2), c(3,6), c(12)),
                                                      ks_all, boundary_all, crossing_all,
                                                      digits_p = 3, digits_pct = 1, digits_e = 2,
                                                      caption = NULL, label = NULL) {
  df_region <- collect_region_calibration(region_key,
                                          horizons = unique(unlist(horizon_blocks)),
                                          ks_all = ks_all,
                                          boundary_all = boundary_all,
                                          crossing_all = crossing_all)
  blocks <- lapply(horizon_blocks, function(hs)
    latex_calibration_block(df_region, hs, digits_p, digits_pct, digits_e))
  
  cap <- if (!is.null(caption)) paste0("  \\caption{", caption, "}\n") else ""
  lab <- if (!is.null(label))   paste0("  \\label{", label, "}\n")   else ""
  
  paste0("\\begin{table}[h]\n  \\centering\n  ",
         "  {\\setlength{\\tabcolsep}{3pt} \\renewcommand{\\arraystretch}{0.85}\n",
         paste(blocks, collapse = "\n  \\vspace{0.6em}\n  "),
         "\n", cap, lab, "\\end{table}\n")
}

# Small helper to write region tables
write_combined_region_table <- function(region_key, label, caption) {
  # Define the structure: h=1/2 together, h=3/6 together, h=12 alone
  horizon_blocks <- list(c(1, 2), c(3, 6), c(12))
  
  # Use the existing render function to generate the full table string
  tex <- render_region_calibration_tables_from_all(
    region_key     = region_key,
    horizon_blocks = horizon_blocks,
    ks_all         = KS_RESULTS_ALL,
    boundary_all   = BOUNDARY_RESULTS_ALL,
    crossing_all   = CROSSING_RESULTS_ALL,
    caption        = caption,
    label          = label
  )
  
  # Write the complete table to a single file
  writeLines(tex, file.path(TABLE_DIR, sprintf("%s_calibration_summary.tex", region_key)))
}
