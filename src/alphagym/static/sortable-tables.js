(() => {
  const collator = new Intl.Collator("zh-CN", { numeric: true, sensitivity: "base" });

  function valueOf(cell, type) {
    const raw = (cell.dataset.sortValue ?? cell.textContent).trim();
    if (!raw || raw === "—") return null;
    if (type === "number") {
      const value = Number(raw.replaceAll(",", "").replace(/%$/, ""));
      return Number.isNaN(value) ? null : value;
    }
    return raw;
  }

  document.querySelectorAll("table[data-sortable]").forEach((table) => {
    const body = table.tBodies[0];
    if (!body) return;
    const headers = Array.from(table.tHead?.rows[0]?.cells ?? []);
    const originalOrder = new Map(Array.from(body.rows).map((row, index) => [row, index]));

    headers.forEach((header, column) => {
      const label = header.textContent.trim();
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sort-button";
      button.textContent = label;
      button.setAttribute("aria-label", `按${label}排序`);
      header.textContent = "";
      header.append(button);

      button.addEventListener("click", () => {
        const ascending = header.getAttribute("aria-sort") !== "ascending";
        headers.forEach((item) => item.removeAttribute("aria-sort"));
        header.setAttribute("aria-sort", ascending ? "ascending" : "descending");
        const type = header.dataset.sortType || "text";
        const rows = Array.from(body.rows);
        rows.sort((left, right) => {
          const a = valueOf(left.cells[column], type);
          const b = valueOf(right.cells[column], type);
          if (a === null && b === null) {
            return originalOrder.get(left) - originalOrder.get(right);
          }
          if (a === null) return 1;
          if (b === null) return -1;
          const compared = type === "number" ? a - b : collator.compare(a, b);
          return ascending ? compared : -compared;
        });
        rows.forEach((row) => body.append(row));
      });
    });
  });
})();
