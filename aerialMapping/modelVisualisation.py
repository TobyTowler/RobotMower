def visualize_predictions(model, dataset, idx, threshold=0.7):
    img, _ = dataset[idx]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])[0]

    # Convert image for display
    img = img.mul(255).permute(1, 2, 0).byte().numpy()

    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Define colors for each class
    colors = ["red", "green", "blue", "yellow", "purple"]

    # Process predictions
    for box, score, label, mask in zip(
        prediction["boxes"],
        prediction["scores"],
        prediction["labels"],
        prediction["masks"],
    ):
        if score > threshold:
            # Get class name and color
            class_name = dataset.class_names[label]
            color = colors[label % len(colors)]

            # Get mask contours
            mask = mask[0].cpu().numpy() > 0.5
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw contours
            for contour in contours:
                contour = contour.reshape(-1, 2)
                plt.plot(contour[:, 0], contour[:, 1], color=color, linewidth=2)

            # Draw bounding box
            x1, y1, x2, y2 = box.cpu().numpy()
            plt.text(
                x1,
                y1,
                f"{class_name}: {score:.2f}",
                bbox=dict(facecolor=color, alpha=0.5),
            )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"prediction_{idx}.png")
    plt.close()
